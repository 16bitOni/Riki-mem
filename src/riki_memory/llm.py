"""Bring your own language model.

The memory system never imports an LLM SDK. It talks to a small :class:`LLMProvider`
protocol, and ships one implementation for any OpenAI-compatible endpoint
(OpenAI, xAI, Ollama, Bedrock-via-proxy, …). Plug in LangChain, LiteLLM or a
local model by implementing the three methods.
"""

import json
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

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


class OpenAICompatibleProvider:
    """``LLMProvider`` over the OpenAI ``/chat/completions`` + ``/embeddings`` HTTP API.

    Works with any OpenAI-compatible server. Structured output uses the
    ``json_schema`` response format, so the server must support it
    (OpenAI, xAI, Ollama and most proxies do).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        chat_model: str = "gpt-4o-mini",
        fast_model: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._fast_model = fast_model or chat_model
        self._embedding_model = embedding_model
        self._timeout = timeout
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

    async def chat_json(self, system: str, user: str, schema: type[M], *, fast: bool = True) -> M:
        client = await self._client_()
        # Pydantic v2 emits a strict JSON schema; most OpenAI-compatible servers accept it.
        json_schema = schema.model_json_schema()
        # `additionalProperties: false` breaks some servers; the schema is advisory, not a contract.
        json_schema.pop("additionalProperties", None)
        response = await client.post(
            "/chat/completions",
            json={
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
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return schema.model_validate(json.loads(content))

    async def chat_text(self, system: str, user: str, *, fast: bool = True) -> str:
        client = await self._client_()
        response = await client.post(
            "/chat/completions",
            json={
                "model": self._fast_model if fast else self._chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
        )
        response.raise_for_status()
        return (response.json()["choices"][0]["message"]["content"] or "").strip()

    async def embed(self, text: str) -> list[float]:
        client = await self._client_()
        response = await client.post(
            "/embeddings", json={"model": self._embedding_model, "input": text}
        )
        response.raise_for_status()
        return [float(x) for x in response.json()["data"][0]["embedding"]]


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
