# riki-memory

A standalone long-term memory layer for AI agents. Extracted from [Riki](https://github.com/16bitOni/Riki) — a personal AI agent — with the rough edges fixed.

**What it does**

- **Extraction** — an LLM turns each conversation exchange into durable, self-contained memories (facts, preferences, events, episodes), with contradiction handling via `replaces`.
- **Hybrid retrieval** — every recall runs four channels in parallel (keyword via Postgres full-text search, semantic via pgvector HNSW, time-window, entity-graph walk) and merges them with weighted Reciprocal Rank Fusion. A failing channel degrades gracefully instead of breaking recall.
- **Dedup with teeth** — content-hash exact match, trigram near-duplicate detection gated on kind + date slot (so "dentist in March" never merges with "dentist in April"), and vector-level dedup restricted to embeddings from the current model.
- **Drift protection** — every vector records which model made it; search never compares vectors across models, and `reembed_stale()` backfills after a model change.
- **Maintenance** — health checks, stale re-embedding, LLM consolidation of paraphrase drift, and history re-extraction.

**What it is not**

- Not a hosted service. It's a library that keeps its own tables in a Postgres database you point it at (pgvector required). Your app's schema is never touched — the memory system is a separate persistence boundary, like Redis.
- Not framework-locked. You bring the LLM via a 3-method `LLMProvider` protocol. An `OpenAICompatibleProvider` (any OpenAI-style `/chat/completions` + `/embeddings` endpoint: OpenAI, xAI, Ollama, proxies) ships in the box.

## Install

```bash
uv add riki-memory
# or: pip install riki-memory
```

You need Postgres with the [pgvector](https://github.com/pgvector/pgvector)
extension. Locally, the repo's `docker-compose.yml` runs one on port 5433:

```bash
docker compose up -d
```

## Quick start

```python
import asyncio
from riki_memory import RikiMemory, OpenAICompatibleProvider, MemoryConfig

async def main():
    memory = RikiMemory(
        database_url="postgresql://riki:riki@localhost:5433/riki",
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
| `forget(user_id, memory_id)` | Hide a memory (kept on record). Final — re-saving won't resurrect it |
| `get` / `list_memories(limit, offset)` | Fetch one memory / newest-first, paged |
| `edit(user_id, memory_id, content=…, importance=…, …)` | Fix a memory. A text change saves a new version and retires the old one |
| `delete(user_id, memory_id)` | **Hard** delete: the memory, its older versions, episode summaries and profile text written from it |
| `delete_user(user_id)` | Delete everything stored for one user |
| `export_user(user_id, path)` / `import_user(path)` | Streaming JSONL backup and restore; import is safe to re-run, `as_user=` makes a copy |
| `health(user_id?)` | `needs_embedding` count, models in use, and `writes`: this process's write stats |
| `reembed_stale(user_id?)` | Backfill vectors after an embedding-model change |
| `consolidate(user_id, dry_run)` | Merge paraphrase drift. Schedule it (weekly cron is plenty) |
| `evaluate(user_id, cases)` | Hit-rate / MRR over your own `EvalCase` JSON file |

## Seeing what the write path does

Every `remember()` produces a `WriteReport`: memories proposed / saved / dropped,
failures and their errors, model calls, tokens, latency. Totals are in
`memory.write_stats`; pass `on_write=` to forward each report to your logs or
metrics:

```python
memory = RikiMemory(database_url=..., llm=..., on_write=lambda r: metrics.emit(r.as_dict()))
```

Alert on `failed_writes`, and on a jump in `empty_writes`: a run of writes that
propose nothing without any error usually means the model's replies aren't
being understood. The library also logs a warning after
`empty_write_warning_after` (default 25) such writes in a row.

## Upgrades

`init()` runs the schema migrations (Alembic, shipped in the package), so
upgrading the library never means re-ingesting. Startups are serialised by a
Postgres lock, the version lives in its own `riki_schema_version` table (no
clash with your app's Alembic), and databases created before migrations
existed are adopted in place.

## Where the data lives

In Postgres, wherever `database_url` points: self-hosted, Neon, RDS, Supabase,
anything with pgvector available. Everything runs inside that one database, so
saving a memory, its countable instances, its entities and its vector is one
transaction — they can never drift apart.

- **Keyword search**: Postgres full-text search (`tsvector` + `ts_rank_cd`, GIN index).
- **Vector search**: pgvector HNSW, cosine distance. Each embedding model gets
  its own partial index, so an old and a new model can coexist while
  `reembed_stale()` catches up; vectors from different models are never compared.
  Indexes are built `CONCURRENTLY`, so a live table keeps taking writes.
- **Counting**: plain SQL over the `instances` table.

```python
memory = RikiMemory(
    # plain postgresql:// URLs are fine; asyncpg is used underneath
    database_url="postgresql://user:pass@ep-abc-123.neon.tech/memorydb?ssl=require",
    llm=...,
)
await memory.init()  # creates the vector extension, tables and indexes (idempotent)
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
9. Postgres + pgvector is the only backend: one transactional store for text, vectors, entities and countable instances.

## Development

```bash
uv sync --group dev
docker compose up -d   # Postgres + pgvector on :5433; each test gets its own database
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
