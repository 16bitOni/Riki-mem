"""The front door. One object, one database, one LLM provider.

Typical wiring in a chat loop::

    memory = RikiMemory(
        database_url="sqlite+aiosqlite:///memory.db",
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
from typing import Awaitable, Callable

from . import extraction, maintenance, recall, store
from .config import MemoryConfig
from .db import Database
from .evaluation import EvalCase, run_eval
from .llm import LLMProvider, NullProvider
from .models import Memory
from .schemas import MemoryCreate

log = logging.getLogger(__name__)


class RikiMemory:
    """Standalone long-term memory for one application."""

    def __init__(
        self,
        database_url: str = "sqlite+aiosqlite:///riki-memory.db",
        llm: LLMProvider | None = None,
        config: MemoryConfig | None = None,
        on_important: Callable[[Memory], Awaitable[None]] | None = None,
    ) -> None:
        """
        - ``database_url``: SQLAlchemy URL. SQLite (``sqlite+aiosqlite:///…``) works
          out of the box; Postgres (``postgresql+asyncpg://…``, e.g. Neon) works
          with the ``postgres`` extra — the keyword engine switches automatically.
          The schema is owned by this package.
        - ``llm``: your model provider. Defaults to :class:`NullProvider`, which
          raises on any model call — fine for ``save``/``forget``/``search``-free
          flows, not for extraction or recall.
        - ``on_important``: async callback fired (awaited) when a memory with
          importance ≥ 4 is saved. Keep it quick, or schedule background work
          from inside it.
        """
        self.db = Database(database_url)
        self.llm: LLMProvider = llm or NullProvider()
        self.config = config or MemoryConfig()
        self.on_important = on_important

    async def init(self) -> None:
        """Create tables. Safe to call on every startup."""
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
        return await extraction.extract_and_save_turn(
            self.db, self.llm, self.config, user_id, user_message, reply,
            known=known, turn_key=turn_key, on_important=self.on_important,
            at=at,
        )

    async def forget(self, user_id: str, memory_id: str) -> bool:
        """Soft-delete a memory. Forgetting is final — re-saving the same
        sentence will not resurrect it."""
        return await store.forget_memory(self.db, self.config, user_id, memory_id)

    async def get(self, user_id: str, memory_id: str) -> Memory | None:
        return await store.get_memory(self.db, memory_id, user_id)

    async def list_memories(
        self, user_id: str, *, active_only: bool = True, limit: int = 100
    ) -> list[Memory]:
        return await store.list_memories(self.db, user_id, active_only=active_only, limit=limit)

    # -- reading -----------------------------------------------------------

    async def recall(
        self, user_id: str, query: str, recent_chat: str = ""
    ) -> tuple[list[Memory], dict]:
        """Plan + hybrid search. Returns ``(memories, report)`` — the report
        carries per-channel hits, timings and scores for debugging."""
        return await recall.search(self.db, self.llm, self.config, user_id, query, recent_chat)

    async def context(
        self, user_id: str, message: str, recent_chat: str = "", name: str = ""
    ) -> tuple[str, list[Memory]]:
        """Prompt-ready context block (profile + relevant memories) for the message."""
        return await recall.build_memory_context(
            self.db, self.llm, self.config, user_id, message, recent_chat, name=name
        )

    # -- maintenance -------------------------------------------------------

    async def health(self, user_id: str | None = None) -> dict:
        return await maintenance.memory_health(self.db, self.llm, user_id)

    async def reembed_stale(self, user_id: str | None = None, limit: int | None = None) -> dict:
        return await maintenance.reembed_stale(self.db, self.llm, self.config, user_id, limit)

    async def consolidate(self, user_id: str, dry_run: bool = False) -> dict:
        """Merge paraphrase drift. Schedule it (weekly cron is plenty)."""
        return await maintenance.consolidate_memories(
            self.db, self.llm, self.config, user_id, dry_run=dry_run
        )

    async def evaluate(self, user_id: str, cases: list[EvalCase]) -> dict:
        return await run_eval(self.db, self.llm, self.config, user_id, cases)
