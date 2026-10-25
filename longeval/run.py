#!/usr/bin/env python
"""Run the LongMemEval benchmark against riki-memory.

Usage:
    # First time: copy .env.example to .env and configure your API key
    cp longeval/.env.example longeval/.env

    # Run the full benchmark (download + ingest + query + report):
    uv run python longeval/run.py

    # Run with options:
    uv run python longeval/run.py --limit 20            # only 20 questions
    uv run python longeval/run.py --judge llm           # LLM judge instead of substring
    uv run python longeval/run.py --skip-ingest         # skip ingestion (reuse existing DB)
    uv run python longeval/run.py --verbose             # show individual misses
    uv run python longeval/run.py --split s              # use the "s" split (~277 MB)
    uv run python longeval/run.py --list-splits          # show available dataset splits
    uv run python longeval/run.py --dataset-path x.json  # use a local dataset file

    # Use a different provider (e.g. xAI):
    #   Set RIKI_BASE_URL=https://api.x.ai/v1 and RIKI_CHAT_MODEL=grok-... in .env
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (
    Dataset,
    aggregate,
    build_memory,
    download_dataset,
    env_get,
    format_report,
    ingest_sessions,
    list_hf_splits,
    load_env,
    parse_dataset,
    run_questions,
)

log = logging.getLogger("longeval")


def progress_bar(done: int, total: int, label: str) -> None:
    pct = done / total if total else 1
    bar_len = 30
    filled = int(bar_len * pct)
    try:
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  {bar} {done}/{total}  {label:<30s}")
    except UnicodeEncodeError:
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  {bar} {done}/{total}  {label:<30s}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LongMemEval benchmark against riki-memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env", default="longeval/.env",
        help="Path to .env file (default: longeval/.env)",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Path to a local dataset JSON file (skips download).",
    )
    parser.add_argument(
        "--split", default="oracle",
        choices=["oracle", "s", "m"],
        help="Dataset split: 'oracle' (~15 MB, fastest), 's' (~277 MB), 'm' (~2.7 GB). Default: oracle.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of questions to evaluate.",
    )
    parser.add_argument(
        "--judge", choices=["substring", "llm", "combined"], default="substring",
        help="Scoring method: 'substring' (fast, free), 'llm' (accurate, costs tokens), "
             "or 'combined' (substring first, LLM fallback for misses).",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip session ingestion (reuse an already-populated database).",
    )
    parser.add_argument(
        "--user-id", default="longeval_user",
        help="User ID for the memory store (default: longeval_user).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print individual misses after the summary.",
    )
    parser.add_argument(
        "--list-splits", action="store_true",
        help="List available dataset splits and exit.",
    )
    parser.add_argument(
        "--save-results", default=None,
        help="Save detailed results to a JSON file.",
    )
    parser.add_argument(
        "--enable-episodes", action="store_true",
        help="Enable the episodic memory layer (generates session summaries during ingestion).",
    )
    parser.add_argument(
        "--retrieval-mode", choices=["single", "agentic"], default="single",
        help="Retrieval strategy: 'single' (one-shot) or 'agentic' (multi-step with reflector).",
    )
    parser.add_argument(
        "--skip-backfill", action="store_true",
        help="Skip category facts backfill (use when table is already populated).",
    )
    parser.add_argument(
        "--full-retrieval", action="store_true",
        help="Save all retrieved contents in results JSON (not just top 5).",
    )
    parser.add_argument(
        "--reingest", action="store_true",
        help="Drop the database and re-ingest all sessions from scratch (use after "
             "improving the extraction prompt).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    env = load_env(args.env)
    hf_token = env_get(env, "HF_TOKEN")

    # -- list splits mode ---------------------------------------------------
    if args.list_splits:
        splits = await list_hf_splits()
        print("Available splits:")
        descs = {"oracle": "evidence sessions only, ~15 MB (fastest)",
                 "s": "~40 sessions/question, ~277 MB",
                 "m": "~500 sessions/question, ~2.7 GB"}
        for name, filename in splits.items():
            print(f"  {name:<8s}  {filename:<40s}  {descs.get(name, '')}")
        return

    # -- load dataset -------------------------------------------------------
    if args.dataset_path:
        print(f"Loading dataset from {args.dataset_path} ...")
        raw = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))
    else:
        print(f"Downloading dataset (split={args.split!r}) ...")
        cached = await download_dataset(split=args.split, hf_token=hf_token)
        raw = json.loads(cached.read_text(encoding="utf-8"))

    dataset: Dataset = parse_dataset(raw, limit=args.limit)
    print(f"  {len(dataset.questions)} questions, {len(dataset.sessions)} unique sessions")

    # -- build memory -------------------------------------------------------
    config_overrides = {}
    if args.enable_episodes:
        config_overrides["enable_episodes"] = True
    if args.retrieval_mode != "single":
        config_overrides["retrieval_mode"] = args.retrieval_mode
    memory = build_memory(env, **config_overrides)

    if args.reingest:
        db_url = env_get(env, "RIKI_DB_URL", "sqlite+aiosqlite:///longeval/eval.db")
        if "sqlite" in db_url:
            db_path = db_url.split("///")[-1] if "///" in db_url else "longeval/eval.db"
            import os
            if os.path.exists(db_path):
                os.remove(db_path)
                print(f"  Removed existing database: {db_path}")
        args.skip_ingest = False
        args.skip_backfill = False

    await memory.init()

    try:
        # -- ingest ---------------------------------------------------------
        if not args.skip_ingest:
            print("\nIngesting sessions ...")
            ingest_report = await ingest_sessions(
                memory, args.user_id, dataset.sessions, progress_fn=progress_bar,
            )
            episodes_msg = f", {ingest_report.get('episodes', 0)} episodes" if ingest_report.get("episodes") else ""
            print(
                f"  done: {ingest_report['ingested']} turn pairs from "
                f"{ingest_report['sessions']} sessions{episodes_msg} in "
                f"{ingest_report['elapsed_s']}s"
                + (f" ({ingest_report['errors']} errors)" if ingest_report["errors"] else "")
            )
        else:
            print("\nSkipping ingestion (--skip-ingest)")

        # -- backfill category facts ----------------------------------------
        if not args.skip_backfill:
            print("\nBackfilling category facts ...")
            facts_saved = await memory.backfill_category_facts(args.user_id)
            if facts_saved:
                print(f"  saved {facts_saved} new category facts")
            else:
                print("  already up-to-date")
        else:
            print("\nSkipping category facts backfill (--skip-backfill)")

        # -- query + score --------------------------------------------------
        print(f"\nEvaluating ({args.judge} judge) ...")
        results = await run_questions(
            memory, args.user_id, dataset.questions,
            judge=args.judge, progress_fn=progress_bar,
        )

        # -- report ---------------------------------------------------------
        report = aggregate(results)
        print()
        print(format_report(report, results, verbose=args.verbose))

        # -- save -----------------------------------------------------------
        if args.save_results:
            out = {
                "report": report,
                "results": [
                    {
                        "question_id": r.question_id,
                        "category": r.category,
                        "question": r.question,
                        "expected": r.expected_answers,
                        "retrieved": r.retrieved_contents,
                        "hit": r.hit,
                        "rr": r.reciprocal_rank,
                        "set_recall": r.set_recall,
                        "answer_accuracy": r.answer_accuracy,
                        **({"all_retrieved": r.all_retrieved_contents}
                           if args.full_retrieval else {}),
                    }
                    for r in results
                ],
            }
            Path(args.save_results).write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(f"\nDetailed results saved to {args.save_results}")

    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
