"""The front door. One object, one database, one LLM provider.

Typical wiring in a chat loop (settings from RIKI_* environment variables)::

    memory = RikiMemory.from_env()

    # before answering:
    context, _ = await memory.context(user_id="user_123", message=text, recent_chat=history)

    # after answering (background task is fine — it never raises):
    await memory.remember(user_id="user_123", user_message=text, reply=answer,
                          turn_key=f"chat/{turn_id}")
"""

import asyncio
import inspect
import logging
import os
import time
from datetime import datetime
from typing import Awaitable, Callable

from . import control, episode, extraction, maintenance, recall, store, telemetry, turns
from .config import MemoryConfig
from .db import Database
from .evaluation import EvalCase, run_eval
from .llm import LLMProvider, NullProvider, OpenAICompatibleProvider
from .models import Memory
from .schemas import MemoryCreate
from .telemetry import WriteReport, WriteStats

log = logging.getLogger(__name__)


def read_env_file(path: str) -> dict[str, str]:
    """``KEY=value`` lines of a .env file (comments and blank lines skipped); {} if absent."""
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("\"'")
    return values


class RikiMemory:
    """Standalone long-term memory for one application."""

    def __init__(
        self,
        database_url: str,
        llm: LLMProvider | None = None,
        config: MemoryConfig | None = None,
        on_important: Callable[[Memory], Awaitable[None]] | None = None,
        on_write: Callable[[WriteReport], Awaitable[None] | None] | None = None,
        schema: str = "riki",
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
        - ``schema``: the Postgres schema riki's tables live in (created if missing),
          so they never clash with your app's tables in the same database.
        """
        self.db = Database(database_url, schema=schema)
        self.llm: LLMProvider = llm or NullProvider()
        self.config = config or MemoryConfig()
        self.on_important = on_important
        self.on_write = on_write
        self.write_stats = WriteStats(empty_warning_after=self.config.empty_write_warning_after)

    @classmethod
    def from_env(cls, *, env: dict[str, str] | None = None, **options) -> "RikiMemory":
        """Build from environment variables (or ``env``), so setup is one line.

        Settings come from the environment, plus a ``.env`` file in the working
        directory if there is one (real environment variables win).

        ============================  ==========================================
        ``RIKI_DATABASE_URL``         Postgres URL (required)
        ``RIKI_API_KEY``              model API key; without it no model is used
        ``RIKI_BASE_URL``             OpenAI-compatible endpoint (default OpenAI)
        ``RIKI_CHAT_MODEL``           main model (default ``gpt-4o-mini``)
        ``RIKI_FAST_MODEL``           cheap model for extraction and planning
        ``RIKI_EMBEDDING_MODEL``      default ``text-embedding-3-small``
        ``RIKI_DB_SCHEMA``            Postgres schema for riki's tables (default ``riki``)
        ``RIKI_RERANK_MODEL``         cross-encoder at ``{base_url}/rerank`` (for ``rerank="cross_encoder"``)
        ============================  ==========================================

        ``options`` go to the constructor (``config=``, ``on_write=``, ...).
        """
        if env is None:
            env = {**read_env_file(".env"), **os.environ}
        url = env.get("RIKI_DATABASE_URL") or env.get("RIKI_DB_URL")
        if not url:
            raise ValueError(
                "Set RIKI_DATABASE_URL, e.g. postgresql://riki:riki@localhost:5433/riki "
                "(`docker compose up -d` in the riki-memory repo runs one)."
            )
        llm = options.pop("llm", None)
        if llm is None and env.get("RIKI_API_KEY"):
            chat_model = env.get("RIKI_CHAT_MODEL", "gpt-4o-mini")
            llm = OpenAICompatibleProvider(
                api_key=env["RIKI_API_KEY"],
                base_url=env.get("RIKI_BASE_URL", "https://api.openai.com/v1"),
                chat_model=chat_model,
                fast_model=env.get("RIKI_FAST_MODEL", chat_model),
                embedding_model=env.get("RIKI_EMBEDDING_MODEL", "text-embedding-3-small"),
                rerank_model=env.get("RIKI_RERANK_MODEL") or None,
            )
        options.setdefault("schema", env.get("RIKI_DB_SCHEMA", "riki"))
        return cls(url, llm=llm, **options)

    async def init(self) -> None:
        """Set up the database (extension, migrations, indexes). Optional: the first
        call that touches the database does it. Safe to call on every startup."""
        await self.db.init()

    async def __aenter__(self) -> "RikiMemory":
        await self.init()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

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
        wait: bool = True,
    ) -> list[Memory]:
        """Extract what is worth remembering from one exchange and save it.

        The exchange is written to the turn log first (``MemoryConfig.store_turns``),
        so a crash mid-extraction leaves a turn that :meth:`process_pending` retries.

        ``known``: memories already shown to the extractor (from :meth:`recall`);
        the extractor may only retire those (``None``: the closest ones are looked
        up). ``turn_key``: a stable id for the turn; the same key twice never
        extracts twice. ``at``: when the exchange happened (pass the original time
        when replaying history so relative dates resolve correctly).
        ``wait=False``: only record the turn and return ``[]`` at once; a worker
        (:meth:`run_worker` or :meth:`process_pending`) extracts it later.
        Never raises on model or extraction failures.
        """
        if name:
            from .users import ensure_user

            await ensure_user(self.db, user_id, name=name)
        if not self.config.store_turns:
            if not wait:
                raise ValueError("remember(wait=False) needs the turn log (MemoryConfig.store_turns=True)")
            saved, _ = await self._extract(user_id, user_message, reply, known, turn_key, at)
            return saved

        turn, claimed = await turns.enqueue(
            self.db, user_id, user_message, reply, key=turn_key, at=at,
            known_ids=None if known is None else [str(m.id) for m in known],
            claim=wait, default_timezone=self.config.default_timezone,
        )
        if turn.status == "done":
            return await turns.memories_of(self.db, turn)  # a retry: no model call
        if not claimed:
            return []  # queued, or another worker has it right now
        return await self._process(turn, known)

    async def _process(self, turn, known: list[Memory] | None = None) -> list[Memory]:
        if known is None:
            known = await turns.known_memories(self.db, turn)
        saved, report = await self._extract(
            turn.user_id, turn.user_text, turn.reply_text, known, turn.key, turn.at
        )
        error = None if report.ok else (report.errors[-1] if report.errors else "extraction failed")
        await turns.finish(self.db, turn.id, error)
        return saved

    async def _extract(self, user_id, user_message, reply, known, turn_key, at):
        async with telemetry.tracking(user_id, "turn", turn_key) as report:
            saved = await extraction.extract_and_save_turn(
                self.db, self.llm, self.config, user_id, user_message, reply,
                known=known, turn_key=turn_key, on_important=self.on_important,
                at=at,
            )
            report.saved = len(saved)
        await self._report_write(report)
        return saved, report

    async def process_pending(self, limit: int = 20, user_id: str | None = None) -> dict:
        """Extract queued turns (from ``remember(wait=False)``, or left by a crash).

        Turns are taken oldest first and processed one at a time, so a user's
        updates are applied in the order they were said. Safe to run in several
        processes at once. Returns ``{"processed": n, "failed": n}``.
        """
        claimed = await turns.claim_turns(self.db, limit, user_id=user_id)
        failed = 0
        for turn in claimed:
            await self._process(turn)
            failed += (await turns.status_of(self.db, turn.id)) == "failed"
        return {"processed": len(claimed), "failed": failed}

    async def run_worker(self, stop: asyncio.Event | None = None, poll_seconds: float = 2.0,
                         batch: int = 20) -> None:
        """Process the turn log until ``stop`` is set: drain it, then poll every
        ``poll_seconds``. Run it as a background task or in its own process."""
        stop = stop or asyncio.Event()
        last_expiry = 0.0
        while not stop.is_set():
            try:
                done = (await self.process_pending(batch))["processed"]
                if time.monotonic() - last_expiry > 3600:  # temporary states, hourly is plenty
                    await self.expire()
                    last_expiry = time.monotonic()
            except Exception:
                log.exception("turn worker iteration failed; retrying")
                done = 0
            if not done:
                try:
                    await asyncio.wait_for(stop.wait(), poll_seconds)
                except asyncio.TimeoutError:
                    pass

    async def expire(self, as_of: datetime | None = None, user_id: str | None = None) -> int:
        """Retire temporary states whose end date has passed (kept as history).
        :meth:`run_worker` does this hourly; call it from a cron job otherwise."""
        return await maintenance.expire_memories(self.db, as_of, user_id)

    async def reextract(self, user_id: str, since: datetime | None = None, dry_run: bool = False) -> dict:
        """Run extraction again over the user's logged turns (after a prompt or model change).
        Already-saved memories are recognised, so running it twice adds nothing."""
        history = await turns.history(self.db, user_id, since)
        return await maintenance.reextract_turns(
            self.db, self.llm, self.config, user_id,
            [{"user": t.user_text, "assistant": t.reply_text, "at": t.at, "key": t.key} for t in history],
            dry_run=dry_run,
        )

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
        self, user_id: str, *, active_only: bool = True, limit: int = 100, offset: int = 0,
        attributes: dict | None = None, source: str | None = None,
    ) -> list[Memory]:
        """Newest first, ``limit`` at a time; page with ``offset``. ``attributes``
        keeps only memories whose attributes contain these key/values; ``source="inferred"``
        lists what was guessed about the user rather than said."""
        return await store.list_memories(
            self.db, user_id, active_only=active_only, limit=limit, offset=offset,
            attributes=attributes, source=source,
        )

    async def edit(self, user_id: str, memory_id: str, **changes) -> Memory | None:
        """Change a memory (``content``, ``importance``, ``kind``, ``tags``, ``attributes``, ``event_at``,
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

    # -- core memory (see profile.py) -------------------------------------

    async def get_blocks(self, user_id: str) -> dict[str, list[str]]:
        """The user's core memory: ``{label: [line, ...]}`` for identity, preferences,
        constraints, people and projects (non-empty blocks only)."""
        from .profile import get_blocks

        return {label: [line["text"] for line in lines] for label, lines in (await get_blocks(self.db, user_id)).items()}

    async def set_block(self, user_id: str, label: str, lines: list[str]) -> None:
        """Replace one block, e.g. as a tool the host agent calls ("remember that I'm
        vegetarian" -> constraints). Lines set here have no source memory."""
        from .profile import BLOCK_LABELS, MAX_LINE_CHARS, MAX_LINES, get_blocks, save_blocks
        from .users import ensure_user

        if label not in BLOCK_LABELS:
            raise ValueError(f"label must be one of {', '.join(BLOCK_LABELS)}")
        await ensure_user(self.db, user_id, timezone=self.config.default_timezone)
        blocks = await get_blocks(self.db, user_id)
        cleaned = [" ".join(t.split())[:MAX_LINE_CHARS] for t in lines if t and t.strip()][-MAX_LINES:]
        blocks[label] = [{"text": t, "from": []} for t in cleaned]
        await save_blocks(self.db, user_id, {k: v for k, v in blocks.items() if v})

    # -- namespaces (see Docs/SCOPE_MODEL.md) -------------------------------

    async def create_namespace(
        self, user_id: str, *, tenant_id: str = "default", kind: str = "user", name: str = "",
        timezone: str | None = None,
    ):
        """Create a memory namespace owned by ``tenant_id``; ``kind`` is user, agent,
        team or org. Idempotent; refuses to move an existing namespace to another tenant."""
        return await control.create_namespace(
            self.db, user_id, tenant_id=tenant_id, kind=kind, name=name,
            timezone_name=timezone or self.config.default_timezone,
        )

    async def list_namespaces(self, tenant_id: str | None = None, **options):
        """Namespaces, optionally one tenant's (``kind``, ``limit``, ``offset`` also accepted)."""
        return await control.list_namespaces(self.db, tenant_id, **options)

    async def delete_tenant(self, tenant_id: str) -> int:
        """Delete every namespace of a tenant and everything in them."""
        return await control.delete_tenant(self.db, tenant_id)

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
        self, user_id: str, message: str, recent_chat: str = "", name: str = "",
        at: datetime | None = None,
    ) -> tuple[str, list[Memory]]:
        """Prompt-ready context block (core memory, computed answers, relevant memories)
        for the message. ``at``: answer as of that moment (default now), for replays."""
        return await recall.build_memory_context(
            self.db, self.llm, self.config, user_id, message, recent_chat, name=name, not_after=at
        )

    # -- maintenance -------------------------------------------------------

    async def health(self, user_id: str | None = None) -> dict:
        """Storage health (vectors, models in use) plus ``writes``: the running
        :class:`WriteStats` of this process."""
        report = await maintenance.memory_health(self.db, self.llm, user_id)
        report["writes"] = self.write_stats.as_dict()
        report["turns"] = await turns.counts(self.db, user_id)
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
