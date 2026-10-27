#!/usr/bin/env python
"""Run the LongMemEval benchmark against riki-memory.

Usage:
    # First time: copy .env.example to .env and configure your API key
    cp longeval/.env.example longeval/.env

    # Run the full benchmark (download + ingest + query + report):
    uv run python longeval/run.py

    # Run with options:
    uv run python longeval/run.py --limit 20            # only 20 questions
    uv run python longeval/run.py --judge qa            # the real score: fixed reader + official judge
    uv run python longeval/run.py --judge llm           # LLM judge over retrieved text (diagnostic)
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
    build_qa_clients,
    aggregate,
    build_memory,
    download_dataset,
    env_get,
    format_report,
    list_hf_splits,
    load_env,
    parse_dataset,
    run_benchmark,
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
        "--split", default="s",
        choices=["oracle", "s", "m"],
        help="Dataset split: 's' (the real test: ~48 sessions, ~245 turns per question; default), "
             "'oracle' (evidence sessions only: a quick smoke test, not a benchmark), 'm' (~2.7 GB).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of questions to evaluate.",
    )
    parser.add_argument(
        "--judge", choices=["qa", "substring", "llm", "combined"], default="substring",
        help="'qa': context() goes to a fixed reader (LongMemEval's prompt, READER_MODEL) and the "
             "answer is graded with LongMemEval's own judge prompts (JUDGE_MODEL, default "
             "zai.glm-5): the number comparable with published results, with cost and "
             "latency. 'substring' / 'llm' / 'combined': retrieval diagnostics only.",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip session ingestion (reuse an already-populated database).",
    )
    parser.add_argument(
        "--user-id", default="longeval",
        help="Prefix for the per-question users: question X is stored as <prefix>:X (default: longeval).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Questions ingested and asked at the same time (default: 8). Lower it if the "
             "model provider rate-limits you.",
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
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request drowns the progress bar

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

    await memory.init()

    if args.reingest:
        from sqlalchemy import text

        from riki_memory.models import Base

        # Empties every riki-memory table in RIKI_DB_URL, nothing else.
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        async with memory.db.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
        print("  Cleared the evaluation database")
        args.skip_ingest = False

    try:
        # -- ingest + query + score (each question in its own user) --------
        print(f"\n{'Querying' if args.skip_ingest else 'Ingesting and querying'} "
              f"{len(dataset.questions)} questions, {args.concurrency} at a time ({args.judge} judge) ...")
        qa = None
        if args.judge == "qa":
            qa = build_qa_clients(env, cache=memory.llm if hasattr(memory.llm, "hits") else None)
            print(f"  reader: {qa.reader_model}   judge: {qa.judge_model}")
        results, ingest_report = await run_benchmark(
            memory, dataset, judge=args.judge, concurrency=args.concurrency,
            skip_ingest=args.skip_ingest, user_prefix=args.user_id,
            progress_fn=progress_bar, qa=qa,
        )
        print(
            f"\n  done in {ingest_report['elapsed_s']}s: {ingest_report['ingested']} turn pairs from "
            f"{ingest_report['sessions']} sessions"
            + (f", {ingest_report['errors']} errors" if ingest_report["errors"] else "")
        )
        cache = memory.llm if hasattr(memory.llm, "hits") else None
        if cache is not None:
            print(f"  model-call cache: {cache.hits} hits, {cache.misses} misses ({cache.root})")
            if cache.failures:
                print(f"\n  WARNING: {cache.failures} model calls failed even after retries, so some turns "
                      "were not remembered. These results are incomplete: lower --concurrency and re-run "
                      "(finished calls replay from the cache).")

        writes = memory.write_stats.as_dict()
        if writes["writes"]:
            n = writes["writes"]
            print(
                f"  write cost per turn: {writes['model_calls'] / n:.2f} model calls, "
                f"{writes['embed_calls'] / n:.2f} embedding calls, "
                f"{writes['prompt_tokens'] / n:.0f} prompt tokens ({writes['cached_tokens'] / n:.0f} cached), "
                f"{writes['completion_tokens'] / n:.0f} completion tokens; "
                f"{writes['skipped_writes']} chitchat turns skipped"
                + ("   (tokens count only calls made this run, not cache replays)" if cache is not None else "")
            )

        # -- report ---------------------------------------------------------
        report = aggregate(results)
        print()
        print(format_report(report, results, verbose=args.verbose))

        # -- save -----------------------------------------------------------
        if args.save_results:
            out = {
                "report": report,
                "write_stats": memory.write_stats.as_dict(),
                "results": [
                    {
                        "question_id": r.question_id,
                        "category": r.category,
                        "question": r.question,
                        "expected": r.expected_answers,
                        "retrieved": r.retrieved_contents,
                        "hit": r.hit,
                        "rr": r.reciprocal_rank,
                        **({"answer": r.answer, "context_tokens": r.context_tokens,
                            "read_model_calls": r.read_model_calls, "recall_ms": round(r.recall_ms)}
                           if r.answer else {}),
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
