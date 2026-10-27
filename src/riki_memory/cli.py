"""``riki``: set up, check and manage a riki-memory database from the terminal.

    riki doctor                       is everything wired up correctly?
    riki config                       every option, its default and what it does
    riki users [--tenant acme]        namespaces (users, agents, teams)
    riki export USER backup.jsonl     everything stored for USER, as JSONL
    riki import backup.jsonl          restore an export (safe to re-run)
    riki delete-user USER             delete everything stored for USER
    riki delete-tenant TENANT         delete every namespace of a tenant
    riki upgrade                      apply schema migrations now
    riki worker                       extract queued turns (remember(wait=False)) until stopped
    riki serve --user me              MCP server over stdio (needs riki-mem[mcp])

Settings come from RIKI_* environment variables (see ``RikiMemory.from_env``),
or from a .env file: ``--env-file path`` (default: ./.env if it exists).
"""

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def load_env_file(path: str | None) -> None:
    """KEY=value lines into os.environ (already-set variables win)."""
    from .service import read_env_file

    candidate = Path(path) if path else Path(".env")
    if not candidate.exists():
        if path:
            sys.exit(f"env file not found: {path}")
        return
    for key, value in read_env_file(str(candidate)).items():
        os.environ.setdefault(key, value)


def _memory():
    from . import RikiMemory

    try:
        return RikiMemory.from_env()
    except ValueError as exc:
        sys.exit(str(exc))


def _confirm(question: str, yes: bool) -> None:
    if yes:
        return
    try:
        answer = input(f"{question} Type 'yes' to continue: ") if sys.stdin.isatty() else None
    except EOFError:
        answer = None
    if answer is None:
        sys.exit("\nrefusing without confirmation: pass --yes")
    if answer.strip().lower() != "yes":
        sys.exit("cancelled")


# -- doctor ---------------------------------------------------------------------


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


async def doctor() -> int:
    """Check every piece without changing anything. Returns the number of failures."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from . import migrate
    from .db import async_url

    failures = 0

    def report(status: str, what: str, detail: str = "") -> None:
        nonlocal failures
        failures += status == FAIL
        print(f"[{status}] {what}" + (f": {detail}" if detail else ""))

    url = os.environ.get("RIKI_DATABASE_URL") or os.environ.get("RIKI_DB_URL")
    if not url:
        report(FAIL, "RIKI_DATABASE_URL", "not set (e.g. postgresql://riki:riki@localhost:5433/riki)")
        return failures
    try:
        engine = create_async_engine(async_url(url))
    except ValueError as exc:
        report(FAIL, "RIKI_DATABASE_URL", str(exc))
        return failures

    try:
        async with engine.connect() as conn:
            version = await conn.scalar(text("SHOW server_version"))
            report(OK, "Postgres reachable", f"version {version}")
            available = await conn.scalar(text(
                "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
            ))
            installed = await conn.scalar(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            if installed:
                recent = _version(installed) >= (0, 8)
                report(OK if recent else WARN, "pgvector installed", f"version {installed}"
                       + ("" if recent else " (0.8+ recommended for filtered search)"))
            elif available:
                report(OK, "pgvector available", f"version {available}, created on first use")
            else:
                report(FAIL, "pgvector", "not available on this server (use the pgvector/pgvector image "
                       "or a host that offers the extension)")
            current = await conn.run_sync(migrate.current)
            head = migrate.head()
            if current == head:
                report(OK, "schema", f"up to date ({head})")
            elif current is None:
                report(OK, "schema", "empty database; created on first use (or run: riki upgrade)")
            elif current in migrate.revisions():
                report(OK, "schema", f"at {current}; upgraded to {head} on first use (or run: riki upgrade)")
            else:
                report(FAIL, "schema", f"at {current}, which this version of riki-memory doesn't know: "
                       "the database was used by a newer version")
    except Exception as exc:
        report(FAIL, "Postgres", f"cannot connect: {exc}")
    finally:
        await engine.dispose()

    if not os.environ.get("RIKI_API_KEY"):
        report(WARN, "model", "RIKI_API_KEY not set: save/list/export work, remember/recall need a model")
        return failures

    from . import RikiMemory
    from .schemas import Extraction

    llm = RikiMemory.from_env().llm
    try:
        started = time.perf_counter()
        vector = await llm.embed("riki doctor check")
        report(OK, "embeddings", f"{llm.embedding_model_id}, {len(vector)} dimensions, "
               f"{(time.perf_counter() - started) * 1000:.0f} ms")
    except Exception as exc:
        report(FAIL, "embeddings", f"{llm.embedding_model_id}: {exc}")
    try:
        started = time.perf_counter()
        reply = await llm.chat_json(
            "Extract long-term memories as JSON. Reply with an object with a `memories` list.",
            "User: My dog is called Biscuit.\nAssistant: Cute name!", Extraction, fast=True,
        )
        took = (time.perf_counter() - started) * 1000
        if reply.memories:
            report(OK, "extraction model", f"structured output parses ({took:.0f} ms)")
        else:
            report(WARN, "extraction model", "answered but extracted nothing from an obvious fact; "
                   "check the model and its structured-output support")
    except Exception as exc:
        report(FAIL, "extraction model", f"reply could not be used: {exc}")
    finally:
        await llm.close()
    return failures


# -- config ---------------------------------------------------------------------


def print_config(as_json: bool) -> None:
    from .config import MemoryConfig

    rows = MemoryConfig.describe()
    if as_json:
        print(json.dumps(rows, indent=2, default=str))
        return
    section = None
    for row in rows:
        if row["section"] != section:
            section = row["section"]
            print(f"\n{section.upper()}")
        limits = row["limits"]
        allowed = (
            "one of " + ", ".join(map(str, limits["choices"])) if "choices" in limits
            else ", ".join(f"{k} {v}" for k, v in limits.items())
        )
        default = repr(row["default"]) if row["default"] != "" else "''"
        print(f"  {row['name']} = {default}" + (f"   ({row['type']}{'; ' + allowed if allowed else ''})"))
        for line in textwrap.wrap(row["description"], 90):
            print(f"      {line}")
    print("\nSet them with MemoryConfig(name=value, ...); MemoryConfig(...).changed() shows your overrides.")


# -- data commands --------------------------------------------------------------

_LABELS = {"entity": "entities", "relation": "relations", "memory": "memories", "episode": "episodes",
           "skipped_links": "links skipped (already present)"}


def _counts(counts: dict) -> str:
    return ", ".join(f"{n} {_LABELS.get(kind, kind)}" for kind, n in counts.items() if n or kind == "memory")


async def users(tenant: str | None, limit: int) -> None:
    async with _memory() as memory:
        rows = await memory.list_namespaces(tenant, limit=limit)
        if not rows:
            print("no namespaces")
        for ns in rows:
            print(f"{ns.id}\ttenant={ns.tenant_id}\tkind={ns.kind}\tcreated={ns.created_at:%Y-%m-%d}")


async def export(user: str, target: str, embeddings: bool, since: str | None) -> None:
    async with _memory() as memory:
        counts = await memory.export_user(
            user, target, include_embeddings=embeddings,
            since=datetime.fromisoformat(since) if since else None,
        )
    print(f"exported {user} to {target}: {_counts(counts)}")


async def import_(source: str, as_user: str | None) -> None:
    async with _memory() as memory:
        counts = await memory.import_user(source, as_user=as_user)
    print(f"imported {source}: {_counts(counts)}")


async def delete_user(user: str, yes: bool) -> None:
    _confirm(f"Delete EVERYTHING stored for {user!r}? This cannot be undone.", yes)
    async with _memory() as memory:
        found = await memory.delete_user(user)
    print(f"deleted {user}" if found else f"no namespace {user!r}")


async def delete_tenant(tenant: str, yes: bool) -> None:
    _confirm(f"Delete every namespace of tenant {tenant!r} and everything in them? This cannot be undone.", yes)
    async with _memory() as memory:
        count = await memory.delete_tenant(tenant)
    print(f"deleted {count} namespaces of {tenant}")


async def upgrade() -> None:
    async with _memory() as memory:
        await memory.init()
    print("schema is up to date")


async def worker() -> None:
    async with _memory() as memory:
        print("processing queued turns; Ctrl+C to stop", file=sys.stderr)
        try:
            await memory.run_worker()
        except asyncio.CancelledError:
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="riki", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=None, help="Load RIKI_* settings from this file (default ./.env).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check database, pgvector, schema and model.")
    p = sub.add_parser("config", help="List every option with its default.")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("users", help="List namespaces.")
    p.add_argument("--tenant")
    p.add_argument("--limit", type=int, default=1000)
    p = sub.add_parser("export", help="Export one user's memory as JSONL.")
    p.add_argument("user")
    p.add_argument("file")
    p.add_argument("--embeddings", action="store_true", help="Include vectors (bigger file, no re-embedding).")
    p.add_argument("--since", help="Only changes after this ISO date-time (incremental backup).")
    p = sub.add_parser("import", help="Import a JSONL export.")
    p.add_argument("file")
    p.add_argument("--as-user", help="Import as a copy under this user id.")
    p = sub.add_parser("delete-user", help="Delete everything stored for one user.")
    p.add_argument("user")
    p.add_argument("--yes", action="store_true", help="Don't ask for confirmation.")
    p = sub.add_parser("delete-tenant", help="Delete every namespace of a tenant.")
    p.add_argument("tenant")
    p.add_argument("--yes", action="store_true")
    sub.add_parser("upgrade", help="Apply schema migrations now.")
    sub.add_parser("worker", help="Extract queued turns until stopped (Ctrl+C).")
    p = sub.add_parser("serve", help="Run an MCP server over stdio for one user's memory.")
    p.add_argument("--user", default=os.environ.get("RIKI_USER_ID", "me"), help="Memory namespace (default: me).")

    args = parser.parse_args(argv)
    load_env_file(args.env_file)
    if args.command == "config":
        print_config(args.json)
    elif args.command == "doctor":
        sys.exit(1 if asyncio.run(doctor()) else 0)
    elif args.command == "users":
        asyncio.run(users(args.tenant, args.limit))
    elif args.command == "export":
        asyncio.run(export(args.user, args.file, args.embeddings, args.since))
    elif args.command == "import":
        asyncio.run(import_(args.file, args.as_user))
    elif args.command == "delete-user":
        asyncio.run(delete_user(args.user, args.yes))
    elif args.command == "delete-tenant":
        asyncio.run(delete_tenant(args.tenant, args.yes))
    elif args.command == "upgrade":
        asyncio.run(upgrade())
    elif args.command == "worker":
        asyncio.run(worker())
    elif args.command == "serve":
        from .mcp_server import serve

        serve(args.user)


if __name__ == "__main__":
    main()
