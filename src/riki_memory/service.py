"""The front door. One object, one database, one LLM provider.

Typical wiring in a chat loop::

    memory = RikiMemory(
        database_url="postgresql://riki:riki@localhost:5433/riki",
        llm=OpenAICompatibleProvider(api_key="…", base_url="https://api.x.ai/v1",
                                     chat_model="grok-4", fast_model="grok-4-fast",
                                     embedding_model="text-embedding-3-small"),
    )
    await memory.init()

    # before answering:
    context, _ = await memory.context(user_id="user_123", message=text, recent_chat=history)

    # after answering (background task is fine — it never raises):
    await memory.remember(user_id="user_123", user_message=text, reply=answer,
                          turn_key=f"chat/{turn_id}")
"""

import logging
from datetime import datetime
from typing import Awaitable, Callable

import inspect

from . import control, episode, extraction, maintenance, recall, store, telemetry
from .config import MemoryConfig
from .db import Database
from .evaluation import EvalCase, run_eval
from .llm import LLMProvider, NullProvider
from .models import Memory
from .schemas import MemoryCreate
from .telemetry import WriteReport, WriteStats

log = logging.getLogger(__name__)


class RikiMemory:
    """Standalone long-term memory for one application."""

    def __init__(
        self,
        database_url: str,
        llm: LLMProvider | None = None,
        config: MemoryConfig | None = None,
        on_important: Callable[[Memory], Awaitable[None]] | None = None,
        on_write: Callable[[WriteReport], Awaitable[None] | None] | None = None,
    ) -> None:
        """
        - ``database_url``: a Postgres URL (``postgresql://…`` or
          ``postgresql+asyncpg://…``; self-hosted, Neon, RDS, Supabase). The
          server needs the pgvector extension available; ``docker-compose.yml``
          runs one locally. The schema is owned by this package.
        - ``llm``: your model provider. Defaults to :class:`NullProvider`, which
          raises on any model call — fine for ``save``/``forget``/``search``-free
          flows, not for extraction or recall.
        - ``on_important``: async callback fired (awaited) when a memory with
          importance ≥ 4 is saved. Keep it quick, or schedule background work
          from inside it.
        - ``on_write``: called (sync or async) with a :class:`WriteReport` after
          every :meth:`remember`: memories proposed,
          saved and dropped, failures, model calls, tokens, latency. Forward it
          to your logs or metrics. Running totals are in :attr:`write_stats`.
        """
        self.db = Database(database_url)
        self.llm: LLMProvider = llm or NullProvider()
        self.config = config or MemoryConfig()
        self.on_important = on_important
        self.on_write = on_write
        self.write_stats = WriteStats(empty_warning_after=self.config.empty_write_warning_after)

    async def init(self) -> None:
        """Create the pgvector extension, tables and indexes. Safe to call on every startup."""
        await self.db.init()

    async def close(self) -> None:
        await self.db.close()
        close = getattr(self.llm, "close", None)
        if callable(close):
            await close()

    # -- writing -----------------------------------------------------------

    async def save(self, data: MemoryCreate) -> Memory:
        """Save one validated memory (dedup, entities and callbacks included)."""
        return await store.save_memory(
            self.db, self.llm, self.config, data, on_important=self.on_important
        )

    async def remember(
        self,
        user_id: str,
        user_message: str,
        reply: str,
        known: list[Memory] | None = None,
        turn_key: str | None = None,
        name: str = "",
        at=None,
    ) -> list[Memory]:
        """Extract what is worth remembering from one exchange and save it.

        ``known``: memories already shown to the extractor (from :meth:`recall`);
        the extractor may only retire those. ``turn_key``: a stable id for the
        turn (prevents double-saving on retries). ``at``: override "now" for
        the extractor prompt (pass the turn's original time when replaying
        history so relative dates resolve correctly). Never raises.
        """
        if name:
            from .users import ensure_user

            await ensure_user(self.db, user_id, name=name)
        async with telemetry.tracking(user_id, "turn", turn_key) as report:
            saved = await extraction.extract_and_save_turn(
                self.db, self.llm, self.config, user_id, user_message, reply,
                known=known, turn_key=turn_key, on_important=self.on_important,
                at=at,
            )
            report.saved = len(saved)
        await self._report_write(report)
        return saved

    async def _report_write(self, report: WriteReport) -> None:
        self.write_stats.add(report)
        if not report.ok:
            log.warning("write for %s failed %d time(s): %s", report.user_id, report.failures, report.errors[-1])
        if self.on_write is None:
            return
        try:
            result = self.on_write(report)
            if inspect.isawaitable(result):
                await result
        except Exception:
            log.exception("on_write callback failed")

    async def close_session(
        self, user_id: str, session_key: str, date=None
    ):
        """Generate an episode summary for a completed session.

        Call after all turns of a session have been ingested. Finds memories
        whose source_key starts with ``session_key``, summarises them, and
        links the episode to its atomic memories. Idempotent.
        """
        return await episode.close_session(
            self.db, self.llm, self.config, user_id, session_key, date=date
        )

    async def forget(self, user_id: str, memory_id: str) -> bool:
        """Soft-delete a memory. Forgetting is final — re-saving the same
        sentence will not resurrect it."""
        return await store.forget_memory(self.db, self.config, user_id, memory_id)

    async def get(self, user_id: str, memory_id: str) -> Memory | None:
        return await store.get_memory(self.db, memory_id, user_id)

    async def list_memories(
        self, user_id: str, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        """Newest first, ``limit`` at a time; page with ``offset``."""
        return await store.list_memories(
            self.db, user_id, active_only=active_only, limit=limit, offset=offset
        )

    async def edit(self, user_id: str, memory_id: str, **changes) -> Memory | None:
        """Change a memory (``content``, ``importance``, ``kind``, ``tags``, ``event_at``,
        ``instances``, ``keep_instances``); see :func:`riki_memory.control.edit_memory`.
        A text change saves a new version and retires the old one. Returns the
        current version, or ``None`` if the memory doesn't exist."""
        return await control.edit_memory(self.db, self.llm, self.config, user_id, memory_id, **changes)

    async def delete(self, user_id: str, memory_id: str) -> int:
        """Delete a memory for real, with its older versions and everything written
        from it (episode summaries, the profile text). Returns rows deleted.
        Use :meth:`forget` to hide a memory but keep it on record."""
        return await control.delete_memory(self.db, user_id, memory_id)

    async def delete_user(self, user_id: str) -> bool:
        """Delete everything stored for one user."""
        return await control.delete_user(self.db, user_id)

    async def export_user(self, user_id: str, target, **options) -> dict:
        """Write a user's memory to JSONL (a path or a text file); see
        :func:`riki_memory.control.export_user` for ``include_embeddings`` and ``since``."""
        return await control.export_user(self.db, user_id, target, **options)

    async def import_user(self, source, *, as_user: str | None = None) -> dict:
        """Load a JSONL export. Safe to re-run; ``as_user`` imports a copy under
        another user id. Vectors from another embedding model are dropped and
        rebuilt by :meth:`reembed_stale`."""
        return await control.import_user(
            self.db, source, as_user=as_user, embedding_model=self.llm.embedding_model_id
        )

    # -- reading -----------------------------------------------------------

    async def recall(
        self, user_id: str, query: str, recent_chat: str = "",
        not_after: datetime | None = None,
    ) -> tuple[list[Memory], dict]:
        """Plan + hybrid search. Returns ``(memories, report)`` — the report
        carries per-channel hits, timings and scores for debugging.

        ``not_after``: exclude memories with event_at after this datetime
        (use the question date so future-dated memories don't pollute results).
        """
        return await recall.search(
            self.db, self.llm, self.config, user_id, query, recent_chat,
            not_after=not_after,
        )

    async def context(
        self, user_id: str, message: str, recent_chat: str = "", name: str = ""
    ) -> tuple[str, list[Memory]]:
        """Prompt-ready context block (profile + relevant memories) for the message."""
        return await recall.build_memory_context(
            self.db, self.llm, self.config, user_id, message, recent_chat, name=name
        )

    # -- maintenance -------------------------------------------------------

    async def health(self, user_id: str | None = None) -> dict:
        """Storage health (vectors, models in use) plus ``writes``: the running
        :class:`WriteStats` of this process."""
        report = await maintenance.memory_health(self.db, self.llm, user_id)
        report["writes"] = self.write_stats.as_dict()
        return report

    async def reembed_stale(self, user_id: str | None = None, limit: int | None = None) -> dict:
        return await maintenance.reembed_stale(self.db, self.llm, self.config, user_id, limit)

    async def consolidate(self, user_id: str, dry_run: bool = False) -> dict:
        """Merge paraphrase drift. Schedule it (weekly cron is plenty)."""
        return await maintenance.consolidate_memories(
            self.db, self.llm, self.config, user_id, dry_run=dry_run
        )

    async def evaluate(self, user_id: str, cases: list[EvalCase]) -> dict:
        return await run_eval(self.db, self.llm, self.config, user_id, cases)
