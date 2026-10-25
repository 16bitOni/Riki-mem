"""Under load a memory must be delayed, not lost: rate limits and server errors are retried."""

import httpx
import pytest

from riki_memory import llm as llm_module
from riki_memory.llm import OpenAICompatibleProvider


def _provider(responses, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        status, body = responses[min(len(calls), len(responses)) - 1]
        return httpx.Response(status, json=body, headers={"retry-after": "0"} if status == 429 else {})

    async def no_sleep(_):
        pass

    monkeypatch.setattr(llm_module.asyncio, "sleep", no_sleep)
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://test/v1", max_retries=3)
    provider._client = httpx.AsyncClient(base_url="http://test/v1", transport=httpx.MockTransport(handler))
    return provider, calls


EMBEDDING = {"data": [{"embedding": [0.1, 0.2]}]}


async def test_rate_limits_are_retried_until_they_succeed(monkeypatch):
    provider, calls = _provider([(429, {}), (503, {}), (200, EMBEDDING)], monkeypatch)
    assert await provider.embed("hi") == [0.1, 0.2]
    assert len(calls) == 3
    await provider.close()


async def test_gives_up_after_max_retries(monkeypatch):
    provider, calls = _provider([(429, {})], monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed("hi")
    assert len(calls) == 4  # first try + 3 retries
    await provider.close()


async def test_a_bad_request_is_not_retried(monkeypatch):
    provider, calls = _provider([(400, {"error": "bad"})], monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed("hi")
    assert len(calls) == 1
    await provider.close()


def test_backoff_honours_retry_after():
    assert llm_module._backoff(0, "7") == 7.0
    assert 0.75 <= llm_module._backoff(0, None) <= 1.25
    assert llm_module._backoff(10, None) <= 37.5
