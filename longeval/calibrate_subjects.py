#!/usr/bin/env python
"""Find the right ``subject_match_threshold`` for the embedding model in .env.

    python longeval/calibrate_subjects.py

Embeds the labelled pairs in ``riki_memory.calibration`` with your configured
embedding model, picks the highest threshold that still keeps 95% of true
members as candidates, then runs the LLM confirm step to report end-to-end
precision and recall. Run it again whenever you change embedding model.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness import build_memory, load_env  # noqa: E402

from riki_memory.calibration import calibrate_subject_threshold, confirmed_scores  # noqa: E402


async def main() -> None:
    env = load_env(Path(__file__).parent / ".env")
    mem = build_memory(env)
    try:
        result = await calibrate_subject_threshold(mem.llm)

        print(f"embedding model: {result.model}")
        print("embeddings alone (candidates above the threshold):")
        print(f"{'threshold':>9}  {'precision':>9}  {'recall':>6}")
        for s in result.scores:
            if round(s.threshold * 100) % 5 == 0 or s is result.best:
                mark = "  <- chosen" if s is result.best else ""
                print(f"{s.threshold:>9.2f}  {s.precision:>9.2f}  {s.recall:>6.2f}{mark}")

        print("\nmembers never shown to the LLM at that threshold:")
        for q, subject, _, sim in sorted(result.missed(), key=lambda m: -m[3]):
            print(f"  {q} -> {subject}: {sim:.3f}")

        final = await confirmed_scores(mem.llm, result)
        print(f"\nwith the LLM confirm step: precision {final.precision:.2f}, recall {final.recall:.2f}")
        print(f"set MemoryConfig(subject_match_threshold={result.best.threshold:.2f}) for {result.model}")
    finally:
        await mem.llm.close()


if __name__ == "__main__":
    asyncio.run(main())
