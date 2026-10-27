<div align="center">

# riki-memory

**Long-term memory for AI agents that you can count on, literally.**

Your agent remembers what users tell it across conversations, answers "how many",
"how much" and "when" from data instead of guesswork, and gives users full
control over what is stored about them.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)
[![Postgres + pgvector](https://img.shields.io/badge/storage-Postgres%20%2B%20pgvector-336791.svg)](#storage-and-upgrades)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

[Why](#why-riki-memory) •
[How it works](#how-it-works) •
[Install](#installation) •
[Quick start](#quick-start) •
[Guide](#usage-guide) •
[CLI](#command-line) •
[Configuration](#configuration) •
[Contributing](#contributing)

</div>

---

## The problem, in one example

A user mentions their plants across three conversations, months apart:

> *July:* "I bought a money plant, a snake plant and a tulsi for the balcony."
> *September:* "My tulsi died, I overwatered it."
> *October:* "Got an areca palm for the living room."

Then asks: **"How many plants do I have right now?"**

Most memory layers hand the model a pile of similar-looking sentences and hope it
counts correctly. riki-memory stores the countable things as rows, applies the rule
"the newest mention decides", and gives the model the answer *with its working*:

```text
## Computed from memory (deterministic; use this instead of counting the memories yourself)
COUNT of plant (status: current/occurred, newest mention decides) = 3
Counted:
- Money Plant | current | 2026-07-27 | memory 4400ff92520f428e90a2360a706d78e1
- Snake Plant | current | 2026-07-27 | memory 4400ff92520f428e90a2360a706d78e1
- Areca Palm | current | 2026-10-05 | memory e7d568b0e8474bf78f24eb598259c66c
Left out:
- Tulsi | former | 2026-09-05 | memory 34277ec39f1c41deb48aff2eab45578a

## Memories that may be relevant to this message
- [fact, 2026-07-27] Sub bought a money plant, a snake plant and a tulsi for the balcony.
- [fact, 2026-10-05] Sub got an areca palm for the living room.
- [fact, 2026-09-05] Sub's tulsi died from overwatering.
```

*(Real output of `memory.context()`.)* The model doesn't count; it reads the count,
and anyone can check what was included and why the tulsi was left out.

---

## Why riki-memory

| | What usually goes wrong | What riki-memory does |
|---|---|---|
| **Counting, sums, dates** | The model counts over retrieved text: repeats get double-counted, sold or planned things get included, results change between runs. | Countable things are stored as rows with a status (`occurred`, `current`, `planned`, `desired`, `former`). Counts, sums and date differences are plain SQL, shown with what was counted and what was left out. |
| **Changing facts** | "Lives in Pune" and "moved to Bangalore" both come back; the model picks one. | A newer fact retires the older one (kept as history), and near-duplicates are merged per date slot. |
| **Silent failures** | Extraction quietly stores nothing and nobody notices for weeks. | Every write reports what it did (proposed, saved, failed, tokens). `riki doctor` checks that your model's replies actually parse. |
| **User control** | Deletion means "hidden"; no export. | Hard delete (with history and derived summaries), full JSONL export/import, delete a whole tenant, all one command. |
| **Infrastructure** | A vector DB, a graph DB and a document store that can drift apart. | One Postgres database. A memory, its vector, its entities and its countable facts are saved in one transaction. |
| **Lock-in** | One model vendor, or a hosted service. | A library. Any OpenAI-compatible model (OpenAI, Anthropic via proxy, xAI, Ollama, Bedrock gateways) or your own 3-method provider. |

---

## Features

- **Memory from conversations**: an LLM turns each exchange into short, self-contained
  memories (facts, preferences, events), with absolute dates ("next Friday" becomes a date).
- **Exact answers**: "how many", "how much", "average", "when", "how long ago" are computed
  from stored data, with evidence and exclusions.
- **Hybrid search**: keyword (Postgres full-text), meaning (pgvector), date window and a
  people/places graph, merged with weighted rank fusion; an optional reranker.
- **Cheap reads and writes**: greetings never reach a model, on either side; most
  questions are answered with no model call and one embedding; each turn is one model call
  with a cacheable prompt and one batched embedding.
- **Nothing lost on a crash**: each exchange is written to a turn log before extraction;
  retries of the same turn cost nothing, failures stay visible, and `reextract()` replays
  history after you improve the prompt or change model.
- **Facts over time**: "moved to Bangalore" retires "lives in Pune" but keeps it as history,
  so "where did I live before?" works; temporary states ("in Goa until Sunday") expire.
- **Said vs guessed**: every memory records whether the user said it, the extractor
  inferred it, or it came from the assistant; a guess never overrules what was said.
- **Core memory**: five short blocks (identity, preferences, constraints, people, projects),
  edited a line at a time; constraints such as allergies are in every context.
- **MCP server**: `riki serve` gives Claude Code, Cursor and other MCP clients the memory as tools.
- **Deduplication with care**: exact, near-duplicate (same kind and date slot only, so
  "dentist in March" never merges with "dentist in April"), and by meaning.
- **Your own data types**: structured `attributes` on every memory, filled by you or by
  the extractor following your instructions, and filterable.
- **Full user control**: edit, forget, hard delete, delete user, delete tenant, JSONL
  export and import.
- **Built for production**: Postgres only, automatic schema migrations, retries with
  backoff on rate limits, per-write telemetry, safe concurrent startups.
- **Ready for companies later**: every memory lives in a namespace with a tenant and a kind
  (user, agent, team, org). See [Docs/SCOPE_MODEL.md](Docs/SCOPE_MODEL.md).

---

## How it works

### Writing: what happens after each turn

```mermaid
flowchart LR
    A["remember(user, message, reply)"] --> B["Extraction<br/>(fast model, JSON)"]
    B --> C{"For each proposed memory"}
    C --> D["Exact duplicate?<br/>(content hash)"]
    D -- yes --> X["Return existing"]
    D -- no --> E["Embed<br/>(outside any transaction)"]
    E --> F["Near-duplicate?<br/>(same kind + date slot)"]
    F --> G[("One transaction:<br/>memory + vector + entities<br/>+ countable instances")]
    G --> H["Retire replaced memories<br/>(kept as history)"]
    G --> T["WriteReport<br/>(telemetry)"]
```

1. **Extraction.** One call to your *fast* model reads the exchange and returns memories as
   JSON: content, kind, importance, date, tags, people and places, and any countable
   **instances** ("bought a snake plant" → `item / plant / snake-plant / current`). It also
   sees what is already known, so it reuses the same keys and subjects.
2. **Deduplication.** The same sentence twice returns the first one. A reworded duplicate
   in the same date slot replaces the older wording. Different days never merge.
3. **Saving.** The memory, its vector, its entity links and its instances are written in a
   single transaction. If the embedding call fails, the memory is still saved and flagged
   so `reembed_stale()` can finish the job.
4. **Reporting.** Each write produces a `WriteReport`; running totals are in `health()`.

### Reading: what happens before each answer

```mermaid
flowchart LR
    Q["context(user, question)"] --> R{"Router"}
    R -- "hi / thanks" --> Z["Nothing to fetch"]
    R -- "how many / how much" --> S["SQL over instances<br/>count · sum · average"]
    R -- "when / how long ago" --> TD["Find both events,<br/>date difference"]
    R -- "anything else" --> P["Plan<br/>(heuristic first, model if needed)"]
    P --> K["Keyword"] & V["Meaning"] & W["Date window"] & GR["Graph"]
    K & V & W & GR --> F["Rank fusion<br/>(+ optional rerank)"]
    S --> O["Prompt-ready context:<br/>computed answer + memories"]
    TD --> O
    F --> O
```

- **Router.** Classifies the question without a model call: small talk, counting,
  temporal, exploratory, or a normal lookup.
- **Counting and sums.** The counted noun ("plants") is matched to stored subjects, by
  word first and, only if no word matches, by meaning with one quick model check that a
  stored subject really is a kind of it (a "cat" is not a "dog"). The newest mention of each
  thing decides its status; repeat mentions of the same amount are counted once.
- **Dates.** "How many days between X and Y" and "how long ago did I X" find both events
  and compute the difference in days, weeks, months or years.
- **Search.** Four channels run in parallel and are merged with weighted reciprocal rank
  fusion; a failing channel is skipped, not fatal. The plan comes from rules
  (`planner_mode="escalate"`), and the model planner is only asked when that finds nothing.
- **Output.** `context()` returns text for your system prompt: the computed answer first
  (with evidence and exclusions), then the profile when useful, then the memories.

---

## Requirements

- **Python** 3.10 or newer
- **PostgreSQL** 13+ with the **pgvector** extension (0.8+ recommended). Any host works:
  Docker, self-hosted, Neon, Supabase, RDS, Cloud SQL.
- **A model endpoint** that speaks the OpenAI API (`/chat/completions` with JSON output,
  and `/embeddings`), or your own provider class.

---

## Installation

**1. Install the package**

```bash
pip install riki-mem            # or: uv add riki-mem
```

From source:

```bash
git clone https://github.com/16bitOni/Riki-mem.git
cd Riki-mem
uv sync                         # or: pip install -e .
```

**2. Start a database** (skip if you already have Postgres with pgvector):

```bash
docker compose up -d            # pgvector/pgvector on port 5433, user/password riki
```

**3. Configure**, in a `.env` file in your working directory or as environment variables:

```bash
RIKI_DATABASE_URL=postgresql://riki:riki@localhost:5433/riki
RIKI_API_KEY=sk-...
RIKI_BASE_URL=https://api.openai.com/v1      # any OpenAI-compatible endpoint
RIKI_CHAT_MODEL=gpt-4o-mini                  # main model
RIKI_FAST_MODEL=gpt-4o-mini                  # extraction, planning, reranking
RIKI_EMBEDDING_MODEL=text-embedding-3-small
RIKI_DB_SCHEMA=riki                          # optional: the Postgres schema riki's tables live in
```

**4. Check the setup:**

```bash
riki doctor
```

```text
[  ok  ] Postgres reachable: version 17.6
[  ok  ] pgvector installed: version 0.8.0
[  ok  ] schema: up to date (0003_memory_attributes)
[  ok  ] embeddings: text-embedding-3-small, 1536 dimensions, 180 ms
[  ok  ] extraction model: structured output parses (900 ms)
```

`doctor` changes nothing. It tells you, before your users do, if the database is
unreachable, pgvector is missing, or your model's JSON replies can't be used.

---

## Quick start

```python
import asyncio
from riki_memory import RikiMemory

async def main():
    async with RikiMemory.from_env() as memory:      # tables are created on first use
        user = "user_123"

        # After a turn: remember what matters (safe as a background task, never raises)
        await memory.remember(
            user,
            "I adopted a beagle last weekend, we named him Biscuit.",
            "Congratulations! How is Biscuit settling in?",
            turn_key="chat-1/turn-1",                # stable id, so retries never duplicate
        )

        # Before answering: what the agent should know for this message
        context, memories = await memory.context(user, "What's my dog called?")
        print(context)

asyncio.run(main())
```

Put `context` into your agent's system prompt; that's the whole integration.

---

## Usage guide

### In a chat loop

```python
context, memories = await memory.context(user_id, message, recent_chat=last_few_turns)
reply = await my_agent.answer(message, system=BASE_PROMPT + "\n\n" + context)

# after replying; pass what recall showed so the extractor may update those memories
asyncio.create_task(memory.remember(
    user_id, message, reply, known=memories, turn_key=f"{chat_id}/{turn_no}",
))
```

When replaying old conversations, pass `at=` (the turn's original time) so relative dates
like "next Friday" resolve correctly.

### Saving directly (no model call)

```python
from riki_memory import MemoryCreate, InstanceItem

await memory.save(MemoryCreate(
    user_id="user_123",
    content="Sub's sister Riya works at Zerodha in Bangalore.",
    kind="fact",          # fact | preference | event | episode
    importance=4,         # 1 trivia ... 5 core identity
))

await memory.save(MemoryCreate(
    user_id="user_123",
    content="Sub bought a snake plant on 27 July 2026.",
    instances=[InstanceItem(kind="item", subject="plant", instance_key="snake-plant",
                            display_name="Snake plant", status="current")],
))
```

### Searching with full detail

```python
memories, report = await memory.recall("user_123", "where does my sister work?")
report["route"]       # lookup | aggregate | temporal | exploratory | chitchat
report["planner"]     # which planner ran (heuristic, llm, ...)
report["channels"]    # hits and timing per search channel
```

### The app's own kinds of data

Every memory has `attributes`: your structured fields, indexed and filterable.

```python
from riki_memory import MemoryConfig, RikiMemory

memory = RikiMemory.from_env(config=MemoryConfig(
    extraction_instructions=(
        "For medications, set type='medication' and put the dose and the times "
        "of day in attributes."
    ),
))

# a conversation like "I take 5 mg of amlodipine every morning" now yields
# attributes={"type": "medication", "dose": "5 mg", "times": ["morning"]}
pills = await memory.list_memories("user_123", attributes={"type": "medication"})
```

You can also set `attributes={...}` yourself in `MemoryCreate`.

### Giving users control over their data

```python
await memory.list_memories("user_123", limit=50, offset=0)        # what is stored, newest first
await memory.edit("user_123", memory_id, content="Lives in Bangalore.")  # new version, old kept as history
await memory.forget("user_123", memory_id)       # hide it; never recalled again
await memory.delete("user_123", memory_id)       # remove it for real, with its history
await memory.delete_user("user_123")             # everything for this user

await memory.export_user("user_123", "user_123.jsonl")   # streaming JSONL, any size
await memory.import_user("user_123.jsonl")                # idempotent restore
await memory.import_user("user_123.jsonl", as_user="user_123-copy")  # an independent copy
```

`delete` also removes episode summaries that quoted the memory and clears the profile text
built from it. Exports include forgotten and superseded memories, so history survives a
move; vectors are left out by default (`include_embeddings=True` keeps them).

### Organisations and shared memory

```python
await memory.create_namespace("acme/alice", tenant_id="acme", kind="user")
await memory.create_namespace("acme/eng-team", tenant_id="acme", kind="team")
await memory.list_namespaces("acme")
await memory.delete_tenant("acme")    # an organisation leaves: everything goes
```

Namespaces created implicitly (by a first save) belong to tenant `"default"`. Shared reads
across namespaces are planned; see [Docs/SCOPE_MODEL.md](Docs/SCOPE_MODEL.md).

### Watching the write path

```python
memory = RikiMemory.from_env(on_write=lambda report: metrics.emit(report.as_dict()))

stats = (await memory.health())["writes"]
stats["failed_writes"], stats["empty_writes"], stats["model_calls"], stats["avg_latency_ms"]
```

Alert on `failed_writes`, and on a jump in `empty_writes`. The library also logs a warning
after 25 writes in a row that propose nothing without any error
(`empty_write_warning_after`), the usual sign that the model's replies aren't understood.

### Background extraction and the turn log

```python
await memory.remember(user_id, message, reply, turn_key=key, wait=False)  # a few ms: just logged

# elsewhere (a task in the same app, or `riki worker` in its own process):
await memory.run_worker()            # or: await memory.process_pending() from a cron job

await memory.reextract(user_id)      # replay the log after a prompt or model change
(await memory.health(user_id))["turns"]   # {"done": 120, "pending": 2, "failed": 0}
```

`remember()` without `wait=False` still extracts right away (and still logs the turn).
A hard `delete()` also blanks the logged text of the turn the memory came from.
`MemoryConfig(store_turns=False)` keeps nothing raw.

### Facts over time

A fact the user changes stays as history (`valid_to`). Normal questions see only what is
true now; "used to", "before I moved", "my old ..." also search history, marked
`(NO LONGER TRUE since ...)` in the context. Counting respects the asked-about window
("how many concerts in March"; "how many plants did I have in August" counts what was
owned then). `run_worker()` expires temporary states hourly; otherwise call `expire()`.

### Core memory

```python
await memory.get_blocks("user_123")    # {"constraints": ["Allergic to peanuts."], ...}
await memory.set_block("user_123", "constraints", ["Allergic to peanuts.", "Vegetarian."])
await memory.list_memories("user_123", source="inferred")   # what was guessed, not said
```

Blocks update in the background from new memories (one fast-model call every
`profile_refresh_after` memories, seeing only what is new). Lines cite their memories, so
deleting a memory removes what was written from it.

### MCP server

```bash
pip install "riki-mem[mcp]"
riki serve --user me      # stdio; add it to your MCP client's config
```

Tools: `recall`, `remember`, `list_memories`, `forget`, `get_core_memory`, `set_core_memory`.

### Maintenance

```python
await memory.reembed_stale()           # after changing embedding model, or a failed embed
await memory.consolidate("user_123")   # merge paraphrase drift (weekly is plenty)
await memory.health("user_123")        # vectors pending, models in use, write stats
```

---

## Command line

The `riki` command reads the same `RIKI_*` settings (or `--env-file path`).

| Command | What it does |
|---|---|
| `riki doctor` | Check database, pgvector, schema version, embeddings and the extraction model. Changes nothing. |
| `riki config [--json]` | Every option with its section, default, allowed values and meaning. |
| `riki users [--tenant acme]` | List namespaces. |
| `riki export USER FILE [--embeddings] [--since ISO]` | Export one user's memory as JSONL (incremental with `--since`). |
| `riki import FILE [--as-user USER]` | Import an export; safe to re-run. |
| `riki delete-user USER [--yes]` | Delete everything for a user (asks first). |
| `riki delete-tenant TENANT [--yes]` | Delete every namespace of a tenant (asks first). |
| `riki upgrade` | Apply schema migrations now (for deploy pipelines). |
| `riki worker` | Extract queued turns (`remember(wait=False)`) until stopped. |
| `riki serve [--user me]` | MCP server over stdio for one user's memory (`riki-mem[mcp]`). |

---

## Configuration

Defaults are meant to work as they are. To see every option:

```bash
riki config
```

```python
from riki_memory import MemoryConfig

MemoryConfig.describe()                      # the same, as data
MemoryConfig(rerank="off").changed()         # {'rerank': 'off'}: only your overrides
```

The options people change most:

| Option | Default | What it controls |
|---|---|---|
| `planner_mode` | `"escalate"` | `heuristic` never calls a model to plan; `llm` always does; `escalate` only when rules find nothing. |
| `rerank` | `"off"` | `"cross_encoder"`: a reranking model (`RIKI_RERANK_MODEL`, served at `{base_url}/rerank`: vLLM, TEI, LiteLLM, Cohere, Jina) re-orders the shortlist in milliseconds. `"llm"`: the fast chat model does it (one extra call per question). |
| `store_turns` | `True` | Keep the turn log (durable writes, `wait=False`, `reextract()`). |
| `recall_limit` | `15` | Memories handed to your agent per question. |
| `profile_injection` | `"when_empty"` | Add the user's profile summary `always`, `never`, or only when nothing else matched. |
| `enable_entities` | `True` | People/places graph (relations like "my sister's husband"). |
| `enable_sql_aggregation` | `True` | Exact counting and sums from stored instances. |
| `enable_episodes` | `False` | Session summaries (call `close_session()` after each session). |
| `retrieval_mode` | `"single"` | `agentic` lets a reflector run follow-up searches. |
| `extraction_instructions` | `""` | Your app's own rules for the extractor. |
| `subject_match_threshold` | `0.40` | How close in meaning a stored subject must be to be considered for a count (re-check with `longeval/calibrate_subjects.py` after changing embedding model). |
| `default_timezone` | `"UTC"` | Timezone for users without one; used to resolve "yesterday". |

### Custom model provider

```python
from riki_memory import RikiMemory

class MyProvider:
    embedding_model_id = "my-embed-1"   # recorded on every vector

    async def chat_json(self, system, user, schema, *, fast=True): ...  # return schema instance
    async def chat_text(self, system, user, *, fast=True): ...          # return str
    async def embed(self, text): ...                                    # return list[float]

memory = RikiMemory(database_url="postgresql://...", llm=MyProvider())
```

`fast=True` calls (extraction, planning, reranking) should go to your cheap model. The
built-in `OpenAICompatibleProvider` already retries rate limits and server errors with
backoff and unwraps the reply formats some models use.

---

## Storage and upgrades

Everything lives in the Postgres database you point it at, in its own **Postgres schema**
(`riki` by default; `RikiMemory(..., schema="...")` or `RIKI_DB_SCHEMA`). Your app can have
its own `users` or `memories` tables in the same database: riki never sees or touches them,
and two apps can keep separate memory in one database by using two schemas. Databases from
before 0.2 (tables in `public`) are moved into the schema on first start, rows intact.

| Table | Holds |
|---|---|
| `users` | Namespaces: id, tenant, kind, name, timezone, core-memory blocks |
| `memories` | Content, kind, source, importance, date, valid until, tags, attributes, vector, history links |
| `turns` | The turn log: each exchange as it arrived, with its extraction status |
| `instances` | Countable things: kind, subject, key, status, date, amount and unit |
| `entities`, `entity_relations`, `memory_entities` | People, places, organisations and how they relate |
| `episodes`, `episode_memories` | Optional session summaries |
| `riki_schema_version` | Migration state (its own name, so it never clashes with your app's Alembic) |

- **Search indexes:** a full-text GIN index for keywords; a pgvector HNSW index per
  embedding model, so an old and a new model can coexist while `reembed_stale()` catches
  up. Vectors from different models are never compared. Indexes are built
  `CONCURRENTLY`, so a live table keeps taking writes.
- **Migrations** run automatically on first use (or `riki upgrade`). Concurrent startups
  are serialised with a Postgres advisory lock, and databases from before migrations
  existed are adopted in place. Upgrading never means re-ingesting.

---

## Performance and limits

Measured on a laptop (Docker Desktop on Windows, about 1.7 ms per database round trip;
expect less on a server network):

| Measurement | Result |
|---|---|
| Vector search, user with 80 memories (50k in the table, 1536 dims) | p50 18 ms, exact top-20 recall 1.000 |
| Vector search, user with 10,000 memories | p50 85 ms, exact top-20 recall 1.000 |
| Write, excluding the model call | about 50 ms per turn |
| Write, model call (Claude Haiku 4.5) | about 1.8 to 2.7 s per turn |

- **Each write** costs one fast-model call and one batched embedding call, plus one
  embedding to find the memories the turn may update. Chitchat turns ("ok thanks") cost
  nothing. The extraction instructions are byte-identical on every call, so providers that
  cache prompt prefixes (OpenAI and compatible) bill them at the cached rate; only the
  exchange, the memories it may update and the related known instances are new tokens.
  `WriteReport.cached_tokens` shows how much was served from the cache.
- **Each read** costs one embedding and, by default, no model call (the planner runs only
  when rules find nothing; rerank is opt-in). The context leaves memory ids out; the
  `Memory` objects you get back have them.
- **Search per user is exact.** Postgres compares that user's vectors directly, which is
  right up to tens of thousands of memories per user. Beyond about 50,000 for a single
  user, see the options in [Docs/FIX_PLAN.md](Docs/FIX_PLAN.md) (half-precision vectors,
  smaller dimensions, partitioning).
- **Extraction quality depends on the fast model.** In a blind comparison on real
  conversations, Claude Haiku 4.5, gpt-oss-120b and DeepSeek V3.2 captured about 87% of key
  facts; smaller models captured 47 to 70%.

---

## Evaluation

riki-memory is measured on its own scenarios first, then against public benchmarks.

- **Scenario suite** ([`evals/`](evals)): 24 personal-agent scenarios (updates, counting,
  sums, plans vs. done, dates, people, "I don't know", password privacy, forget and delete,
  noise, user isolation), replayed through the public API with a fixed reader and judge.

  ```bash
  python evals/run.py --dry-run          # check the scenarios, no model calls
  python evals/run.py --verbose          # run them
  python evals/run.py --ablation         # each optional feature on and off, side by side
  ```

- **Public benchmarks** (LongMemEval, LoCoMo, BEAM, HaluMem): the plan, and why vendor
  scores can't be compared with each other, are in [Docs/BENCHMARK.md](Docs/BENCHMARK.md).
  No benchmark score is published yet; numbers will come with the harness to reproduce them.

---

## Project layout

```text
src/riki_memory/
  service.py        RikiMemory: the public API
  extraction.py     conversation -> memories (the extraction prompt)
  store.py          saving, deduplication, forgetting
  recall.py         router, planning, context building
  retrieval.py      the four search channels and rank fusion
  aggregation.py    exact counts, sums and averages over instances
  temporal.py       date differences between remembered events
  control.py        edit, delete, export/import, namespaces
  turns.py          the turn log and its queue
  profile.py        core memory blocks
  mcp_server.py     the MCP server (`riki serve`)
  telemetry.py      per-write reports and totals
  vectors.py        pgvector search and per-model indexes
  migrate.py        schema migrations (migrations/ holds the revisions)
  cli.py            the `riki` command
  config.py         every option, with defaults and descriptions
evals/              our own scenario suite and runner
longeval/           LongMemEval harness (with a model-call cache)
tests/              unit and integration tests (real Postgres)
Docs/               design notes: review, fix plan, benchmarks, scope model
```

---

## Development

```bash
git clone https://github.com/16bitOni/Riki-mem.git && cd Riki-mem
uv sync --group dev
docker compose up -d            # Postgres + pgvector on :5433
uv run pytest                   # every test gets its own fresh database
```

Point the tests at another server with `RIKI_TEST_PG_URL` (the user needs `CREATEDB`).

Changing a model in `models.py`? Add a migration, or `tests/test_migrations.py` will fail:

```bash
python -m riki_memory.migrate revision "describe the change" postgresql://.../scratch_db
```

Build and publish:

```bash
uv build
uv publish
```

---

## Contributing

Contributions are welcome: bug reports, scenarios for the eval suite, providers, docs.

1. Open an issue describing the problem or idea before large changes.
2. Keep the library free of benchmark-specific behaviour: evaluation code lives in
   `evals/` and `longeval/` and uses only the public API.
3. Add or update tests; `uv run pytest` must pass against Postgres.
4. Schema changes come with a migration.

---

## Project status

**Alpha (0.2.0).** The storage layer, write path, exact counting, user data controls and
migrations, turn log, validity over time and core memory are tested (289 tests against
real Postgres). Still in progress:

- Feature pruning based on the scenario suite's ablation results
- Reading across shared namespaces (team / org memory)
- Benchmark harness with a fixed reader and official judges, and published results
- A hosted-service wrapper (HTTP API, auth) on top of the library

---

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Extracted from [Riki](https://github.com/16bitOni/Riki), a personal AI agent, and rebuilt
for production use. Built on [PostgreSQL](https://www.postgresql.org/),
[pgvector](https://github.com/pgvector/pgvector), [SQLAlchemy](https://www.sqlalchemy.org/),
[Alembic](https://alembic.sqlalchemy.org/) and [Pydantic](https://docs.pydantic.dev/).
