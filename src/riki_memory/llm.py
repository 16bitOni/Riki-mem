"""Bring your own language model.

The memory system never imports an LLM SDK. It talks to a small :class:`LLMProvider`
protocol, and ships one implementation for any OpenAI-compatible endpoint
(OpenAI, xAI, Ollama, Bedrock-via-proxy, …). Plug in LangChain, LiteLLM or a
local model by implementing the three methods.
"""

import asyncio
import json
import logging
import random
import re
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

from . import telemetry

M = TypeVar("M", bound=BaseModel)


class LLMProvider(Protocol):
    """What the memory system needs from a language model."""

    async def chat_json(self, system: str, user: str, schema: type[M], *, fast: bool = True) -> M:
        """Answer with JSON validated against ``schema``. ``fast=True`` may use a cheaper model."""
        ...

    async def chat_text(self, system: str, user: str, *, fast: bool = True) -> str:
        """Answer with plain text."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...

    @property
    def embedding_model_id(self) -> str:
        """Name of the embedding model, stored on each memory for drift protection."""
        ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_json_reply(content: str | None, schema: type[BaseModel]) -> object:
    """The JSON object a model meant to send, undoing the usual wrappers.

    - Markdown fences (```json ... ```).
    - The ``json_schema`` name used as a wrapper key: DeepSeek and Llama answer
      ``{"answer": "{...}"}`` (sometimes with the inner object as a string)
      instead of the object itself. Unwrapped only when the outer object has
      none of the schema's own fields, so a real field named "answer" is safe.
    """
    data = json.loads(_FENCE_RE.sub("", (content or "").strip()))
    fields = set(schema.model_fields)
    for _ in range(2):
        if isinstance(data, dict) and len(data) == 1 and not fields & set(data):
            data = next(iter(data.values()))
            if isinstance(data, str):
                data = json.loads(_FENCE_RE.sub("", data.strip()))
    return data


log = logging.getLogger(__name__)

# Worth another try: rate limits, overload, and gateway errors.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class OpenAICompatibleProvider:
    """``LLMProvider`` over the OpenAI ``/chat/completions`` + ``/embeddings`` HTTP API.

    Works with any OpenAI-compatible server. Structured output uses the
    ``json_schema`` response format, so the server must support it
    (OpenAI, xAI, Ollama and most proxies do).

    Rate limits (429), overload and network errors are retried with
    exponential backoff (honouring ``Retry-After``), up to ``max_retries``
    times: under load a memory must be delayed, not silently lost.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        chat_model: str = "gpt-4o-mini",
        fast_model: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        timeout: float = 120.0,
        max_retries: int = 6,
        rerank_model: str | None = None,
    ) -> None:
        """``rerank_model``: a cross-encoder served at ``{base_url}/rerank`` (the
        Cohere/Jina format that vLLM, TEI, LiteLLM and those APIs speak), used when
        ``MemoryConfig.rerank = "cross_encoder"``."""
        self._rerank_model = rerank_model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._fast_model = fast_model or chat_model
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    @property
    def embedding_model_id(self) -> str:
        return self._embedding_model

    async def _client_(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, path: str, body: dict) -> dict:
        client = await self._client_()
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(path, json=body)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt == self._max_retries:
                    raise
                wait = _backoff(attempt, None)
                log.warning("%s failed (%s); retry %d in %.1fs", path, type(exc).__name__, attempt + 1, wait)
            else:
                if response.status_code not in RETRYABLE_STATUS or attempt == self._max_retries:
                    response.raise_for_status()
                    data = response.json()
                    if path == "/embeddings":
                        telemetry.record_embed_call()
                    elif path == "/rerank":
                        pass
                    else:
                        usage = data.get("usage") or {}
                        telemetry.record_model_call(
                            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                        )
                    return data
                wait = _backoff(attempt, response.headers.get("retry-after"))
                log.warning("%s returned %d; retry %d in %.1fs", path, response.status_code, attempt + 1, wait)
            await asyncio.sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover

    async def chat_json(self, system: str, user: str, schema: type[M], *, fast: bool = True) -> M:
        # Pydantic v2 emits a strict JSON schema; most OpenAI-compatible servers accept it.
        json_schema = schema.model_json_schema()
        # `additionalProperties: false` breaks some servers; the schema is advisory, not a contract.
        json_schema.pop("additionalProperties", None)
        if getattr(schema, "compact_schema", False):
            json_schema = _without_descriptions(json_schema)
        data = await self._post(
            "/chat/completions",
            {
                "model": self._fast_model if fast else self._chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": json_schema, "strict": False},
                },
            },
        )
        content = data["choices"][0]["message"]["content"]
        return schema.model_validate(parse_json_reply(content, schema))

    async def chat_text(self, system: str, user: str, *, fast: bool = True) -> str:
        data = await self._post(
            "/chat/completions",
            {
                "model": self._fast_model if fast else self._chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
        )
        return (data["choices"][0]["message"]["content"] or "").strip()

    async def embed(self, text: str) -> list[float]:
        data = await self._post("/embeddings", {"model": self._embedding_model, "input": text})
        return [float(x) for x in data["data"][0]["embedding"]]

    async def rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        """Relevance of each document to ``query`` from the cross-encoder (higher is better)."""
        if not self._rerank_model:
            raise RuntimeError("no rerank_model configured (RIKI_RERANK_MODEL)")
        data = await self._post("/rerank", {"model": self._rerank_model, "query": query, "documents": documents})
        scores = [0.0] * len(documents)
        for row in data.get("results") or data.get("data") or []:
            scores[row["index"]] = float(row.get("relevance_score", row.get("score", 0.0)))
        return scores

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """One request for several texts (the OpenAI embeddings API takes a list)."""
        if not texts:
            return []
        data = await self._post("/embeddings", {"model": self._embedding_model, "input": list(texts)})
        rows = sorted(data["data"], key=lambda row: row.get("index", 0))
        return [[float(x) for x in row["embedding"]] for row in rows]


async def embed_texts(llm, texts: list[str]) -> list[list[float]]:
    """Embed several texts: one batched call when the provider has ``embed_many``,
    otherwise ``embed`` per text, in parallel (custom providers need not implement it)."""
    if not texts:
        return []
    many = getattr(llm, "embed_many", None)
    if callable(many):
        return await many(list(texts))
    return list(await asyncio.gather(*(llm.embed(t) for t in texts)))


def _without_descriptions(node):
    """The schema minus descriptions and titles (structure and allowed values stay)."""
    if isinstance(node, dict):
        return {k: _without_descriptions(v) for k, v in node.items() if k not in ("description", "title")}
    if isinstance(node, list):
        return [_without_descriptions(v) for v in node]
    return node


def _backoff(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait: the server's ``Retry-After`` if it gave one, else 1, 2, 4 ... 30 with jitter."""
    if retry_after:
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(30.0, 2.0 ** attempt) * random.uniform(0.75, 1.25)


class NullProvider:
    """An ``LLMProvider`` that refuses to call any model.

    Useful for tests and for deployments that only use :meth:`save` /
    :meth:`search` with pre-computed data. Any LLM call raises ``RuntimeError``.
    """

    embedding_model_id = "null"

    async def chat_json(self, system: str, user: str, schema: type[M], *, fast: bool = True) -> M:
        raise RuntimeError("NullProvider cannot answer; plug in a real LLMProvider.")

    async def chat_text(self, system: str, user: str, *, fast: bool = True) -> str:
        raise RuntimeError("NullProvider cannot answer; plug in a real LLMProvider.")

    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("NullProvider cannot embed; plug in a real LLMProvider.")
