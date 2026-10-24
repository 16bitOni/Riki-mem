# riki-memory

A standalone long-term memory layer for AI agents. Extracted from [Riki](https://github.com/16bitOni/Riki) — a personal AI agent — with the rough edges fixed.

**What it does**

- **Extraction** — an LLM turns each conversation exchange into durable, self-contained memories (facts, preferences, events, episodes), with contradiction handling via `replaces`.
- **Hybrid retrieval** — every recall runs four channels in parallel (keyword via SQLite FTS5, semantic via cosine similarity, time-window, entity-graph walk) and merges them with weighted Reciprocal Rank Fusion. A failing channel degrades gracefully instead of breaking recall.
- **Dedup with teeth** — content-hash exact match, trigram near-duplicate detection gated on kind + date slot (so "dentist in March" never merges with "dentist in April"), and vector-level dedup restricted to embeddings from the current model.
- **Drift protection** — every vector records which model made it; search never compares vectors across models, and `reembed_stale()` backfills after a model change.
- **Maintenance** — health checks, stale re-embedding, LLM consolidation of paraphrase drift, and history re-extraction.

**What it is not**

- Not a hosted service. It's a library with its own SQLite database (WAL mode, FTS5). Your app's schema is never touched — the memory system is a separate persistence boundary, like Redis.
- Not framework-locked. You bring the LLM via a 3-method `LLMProvider` protocol. An `OpenAICompatibleProvider` (any OpenAI-style `/chat/completions` + `/embeddings` endpoint: OpenAI, xAI, Ollama, proxies) ships in the box.

## Install

```bash
uv add riki-memory
# or: pip install riki-memory
```

## Quick start

```python
import asyncio
from riki_memory import RikiMemory, OpenAICompatibleProvider, MemoryConfig

async def main():
    memory = RikiMemory(
        database_url="sqlite+aiosqlite:///memory.db",
        llm=OpenAICompatibleProvider(
            api_key="...",
            base_url="https://api.x.ai/v1",   # any OpenAI-compatible endpoint
            chat_model="grok-4",
            fast_model="grok-4-fast",          # cheap model for planning/extraction
            embedding_model="text-embedding-3-small",
        ),
        config=MemoryConfig(),
        on_important=lambda m: print(f"important: {m.content}"),  # optional
    )
    await memory.init()

    # Before answering: get prompt-ready context for this message.
    context, memories = await memory.context(
        user_id="user_123", message="where does my sister work?", recent_chat="",
    )

    # After answering (run it in the background — it never raises):
    await memory.remember(
        user_id="user_123",
        user_message="where does my sister work?",
        reply="Riya works at Zerodha.",
        known=memories,          # what recall showed, so `replaces` stays honest
        turn_key="chat/turn-42", # stable id: safe to retry
    )

    await memory.close()

asyncio.run(main())
```

Direct writes skip the LLM entirely:

```python
from riki_memory import MemoryCreate

await memory.save(MemoryCreate(
    user_id="user_123",
    content="Sub's sister Riya works at Zerodha in Bangalore.",
    kind="fact",
    importance=4,
))
```

## API

| Method | What it does |
|---|---|
| `context(user_id, message, recent_chat)` | Prompt-ready block: profile + relevant memories |
| `recall(user_id, query, recent_chat)` | `(memories, report)` — report has per-channel hits, timings, scores |
| `remember(user_id, user_message, reply, …)` | Extract + save + refresh profile; never raises |
| `save(MemoryCreate)` | Direct validated write (dedup, entities, callbacks) |
| `forget(user_id, memory_id)` | Soft delete. Final — re-saving won't resurrect it |
| `get` / `list_memories` | Fetch one memory / newest-first listing |
| `health(user_id?)` | `needs_embedding` count, models in use |
| `reembed_stale(user_id?)` | Backfill vectors after an embedding-model change |
| `consolidate(user_id, dry_run)` | Merge paraphrase drift. Schedule it (weekly cron is plenty) |
| `evaluate(user_id, cases)` | Hit-rate / MRR over your own `EvalCase` JSON file |

## Where the data lives

Wherever `database_url` points. SQLite is the default (a local file, WAL mode,
FTS5 keyword index) — fine for personal agents and for shipping. Nothing about
the design is local-only: point it at Postgres and the keyword engine switches
automatically from FTS5 to `tsvector` + `ts_rank_cd` (GIN-indexed); dedup,
vector search, the graph walk and maintenance are already portable.

```bash
uv add "riki-memory[postgres]"
```

```python
memory = RikiMemory(
    # Neon, RDS, self-hosted Postgres — any SQLAlchemy URL
    database_url="postgresql+asyncpg://user:pass@ep-abc-123.neon.tech/memorydb?ssl=require",
    llm=...,
)
await memory.init()  # creates tables + the tsvector GIN index (idempotent)
```

The schema stays inside the library's own tables in that database — your
application's tables are never touched.

## Token-efficient retrieval

The biggest LLM cost in most memory systems is the *planner* — a model call on
every message just to decide what to search for. `riki-memory` can skip it:

```python
config = MemoryConfig(
    planner_mode="escalate",  # "llm" | "heuristic" | "escalate"
)
```

- `"heuristic"` — keywords, entities and dates come from regex over the message
  (stopword filtering, capitalised-name detection, relative date parsing like
  "last week" or "in March"). **Zero chat tokens**, always.
- `"escalate"` (default) — tries the heuristic plan first and runs the full
  hybrid search; only when it finds nothing does it call the LLM planner (which
  is better at resolving references like "that role" from recent chat). Common
  case: zero tokens. The report tells you which planner ran
  (`report["planner"]`).
- `"llm"` — the original behaviour: always plan with the fast model.

Every mode still makes exactly one embedding call for the semantic channel —
the same "one embedding per query, no LLM in retrieval" shape the reference
systems use. Greetings and small talk never reach any model via the
heuristic pre-gate.

## Configuration

```python
from riki_memory import MemoryConfig

config = MemoryConfig(
    enable_entities=False,      # skip the entity graph (write-heavy at small scale)
    profile_injection="when_empty",  # "always" | "when_empty" | "never"
    recall_limit=8,
    weight_semantic=1.0,
    weight_keyword=1.0,
    weight_graph=0.7,
    weight_time=0.5,
    rerank="llm",               # optional LLM re-ordering of the shortlist
)
```

See `src/riki_memory/config.py` for every tunable and its default.

## Custom LLM provider

```python
from riki_memory import LLMProvider

class MyProvider(LLMProvider):
    embedding_model_id = "my-embed-1"

    async def chat_json(self, system, user, schema, *, fast=True): ...
    async def chat_text(self, system, user, *, fast=True): ...
    async def embed(self, text): ...
```

`fast=True` calls are planning/extraction/rerank — route them to your cheap model.

## What changed vs the Riki original

While extracting, the flaws found in the codebase teardown were fixed:

1. The recall planner runs on the **fast** model, and greetings/thanks/farewells skip the planner entirely via a heuristic pre-gate (the original burned a main-model call on every "hi").
2. **All** kinds are embedded by default — the original left facts invisible to the semantic channel.
3. Exact duplicates no longer **resurrect** forgotten memories.
4. The exact-hash check → insert race is caught: concurrent identical saves return the same row instead of crashing.
5. The entity graph is **optional** (`enable_entities=False` skips its write overhead and the graph channel).
6. Consolidation is a first-class `consolidate()` method, not a CLI-only script.
7. The hard-coded cognition trigger is now an awaited `on_important` callback the host supplies.
8. Profile injection defaults to `when_empty` (~350 tokens saved on most turns), and the staleness check reads fresh state instead of a request-scoped object.
9. Postgres-only machinery (`pg_trgm`, `tsvector`, `pgvector`/HNSW) replaced with portable equivalents: Python trigram similarity, FTS5 + BM25, brute-force cosine over per-user candidates.

## Development

```bash
uv sync --group dev
uv run pytest
uv build   # dist/riki_memory-*.whl
```

## Publishing to PyPI

```bash
uv build
uv publish   # needs a PyPI token: uv publish --token pypi-…
```

## License

MIT
