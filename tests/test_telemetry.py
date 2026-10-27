"""Every write reports what happened, so a write path that quietly saves nothing gets noticed."""

import asyncio
import json
import logging

import httpx

from riki_memory import MemoryConfig, RikiMemory
from riki_memory.llm import OpenAICompatibleProvider


def _server(extraction_replies, usage=(1000, 50)):
    """A fake OpenAI-compatible server: extraction replies in order, fixed embeddings."""
    replies = list(extraction_replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0, 0.0]}]})
        body = json.loads(request.content)
        if "response_format" not in body:  # a profile refresh, not an extraction
            return httpx.Response(200, json={"choices": [{"message": {"content": "A profile."}}]})
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "nope"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(reply)}}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        })

    provider = OpenAICompatibleProvider(api_key="k", base_url="http://fake/v1", max_retries=0)
    provider._client = httpx.AsyncClient(base_url="http://fake/v1", transport=httpx.MockTransport(handler))
    return provider


ONE = {"memories": [{"content": "Sub adopted a dog named Bruno.", "kind": "fact", "semantic": True}]}
NONE = {"memories": []}


async def _mem(db_url, provider, **kwargs):
    config = kwargs.pop("config", MemoryConfig(planner_mode="heuristic", rerank="off"))
    mem = RikiMemory(database_url=db_url, llm=provider, config=config, **kwargs)
    await mem.init()
    return mem


async def test_a_successful_write_reports_what_it_did(db_url):
    reports = []
    mem = await _mem(db_url, _server([ONE]), on_write=reports.append)
    try:
        await mem.remember("u1", "I adopted a dog named Bruno.", "Congrats!", turn_key="t1")
        (report,) = reports
        assert (report.ok, report.mode, report.source_key) == (True, "turn", "t1")
        assert (report.proposed, report.saved, report.failures) == (1, 1, 0)
        assert (report.model_calls, report.prompt_tokens, report.completion_tokens) == (1, 1000, 50)
        assert report.embed_calls >= 1 and report.latency_ms > 0
        stats = (await mem.health())["writes"]
        assert (stats["writes"], stats["memories_saved"], stats["failed_writes"]) == (1, 1, 0)
    finally:
        await mem.close()


async def test_a_failed_write_is_counted_not_hidden(db_url, caplog):
    seen = []

    async def on_write(report):  # async callbacks are awaited
        seen.append(report)

    mem = await _mem(db_url, _server([400]), on_write=on_write)
    try:
        assert await mem.remember("u1", "I adopted a dog.", "Nice") == []  # still never raises
        (report,) = seen
        assert (report.ok, report.failures, report.saved) == (False, 1, 0)
        assert "HTTPStatusError" in report.errors[0]
        assert mem.write_stats.failed_writes == 1 and "HTTPStatusError" in mem.write_stats.last_error
    finally:
        await mem.close()


async def test_a_run_of_empty_writes_raises_a_warning(db_url, caplog):
    config = MemoryConfig(planner_mode="heuristic", rerank="off", empty_write_warning_after=3)
    mem = await _mem(db_url, _server([NONE]), config=config)
    try:
        with caplog.at_level(logging.WARNING, logger="riki_memory.telemetry"):
            for i in range(3):
                await mem.remember("u1", f"message {i}", "ok")
        assert mem.write_stats.empty_writes == 3
        assert any("in a row proposed no memories" in r.message for r in caplog.records)
    finally:
        await mem.close()


async def test_a_broken_callback_never_breaks_the_write(db_url):
    def explode(report):
        raise RuntimeError("metrics backend down")

    mem = await _mem(db_url, _server([ONE]), on_write=explode)
    try:
        saved = await mem.remember("u1", "I adopted a dog named Bruno.", "Congrats!")
        assert len(saved) == 1
    finally:
        await mem.close()


async def test_concurrent_writes_keep_their_own_numbers(db_url):
    reports = []
    mem = await _mem(db_url, _server([ONE, NONE]), on_write=reports.append)
    try:
        await asyncio.gather(
            mem.remember("u1", "I adopted a dog named Bruno.", "Congrats!", turn_key="a"),
            mem.remember("u2", "what's a good name for a cat?", "Mochi.", turn_key="b"),
        )
        assert sorted(r.model_calls for r in reports) == [1, 1]
        assert sum(r.proposed for r in reports) == 1
        assert mem.write_stats.writes == 2
    finally:
        await mem.close()


async def test_chitchat_turns_cost_nothing_and_are_not_empty_writes(db_url):
    reports = []
    mem = await _mem(db_url, _server([ONE]), on_write=reports.append)
    try:
        assert await mem.remember("u1", "ok thanks!", "Anytime.") == []
        assert reports[0].skipped and reports[0].model_calls == 0 and reports[0].embed_calls == 0
        assert mem.write_stats.skipped_writes == 1 and mem.write_stats.empty_writes == 0
    finally:
        await mem.close()


async def test_the_cross_encoder_reorders_with_one_rerank_request(db_url):
    """rerank="cross_encoder": one /rerank request, no chat call, and the best-scored memory first."""
    import httpx as _httpx

    from riki_memory import MemoryCreate

    seen = []

    def handler(request):
        if request.url.path.endswith("/embeddings"):
            texts = json.loads(request.content)["input"]
            texts = texts if isinstance(texts, list) else [texts]
            axes = ("Paris", "Rome", "Lyon")  # distinct vectors, so the three never merge as duplicates
            vectors = [[1.0 if a in t else 0.0 for a in axes] if any(a in t for a in axes) else [1.0, 1.0, 1.0]
                       for t in texts]
            return _httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]})
        assert request.url.path.endswith("/rerank"), "no chat call expected"
        body = json.loads(request.content)
        seen.append(body)
        scores = [1.0 if "Lyon" in doc else 0.1 for doc in body["documents"]]
        return _httpx.Response(200, json={"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]})

    provider = OpenAICompatibleProvider(api_key="k", base_url="http://fake/v1", max_retries=0, rerank_model="bge-reranker")
    provider._client = _httpx.AsyncClient(base_url="http://fake/v1", transport=_httpx.MockTransport(handler))
    mem = RikiMemory(database_url=db_url, llm=provider,
                     config=MemoryConfig(planner_mode="heuristic", rerank="cross_encoder"))
    try:
        for text in ("Riya's sister moved to Paris.", "Riya visited Rome.", "Riya lives in Lyon now."):
            await mem.save(MemoryCreate(user_id="u1", content=text, kind="fact"))
        found, report = await mem.recall("u1", "Where does Riya live?")
        assert report["reranked"] and found[0].content == "Riya lives in Lyon now."
        assert len(seen) == 1 and seen[0]["model"] == "bge-reranker"
    finally:
        await mem.close()
