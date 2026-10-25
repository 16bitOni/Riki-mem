"""On-disk cache for model calls, so benchmark re-runs cost nothing.

Every request is made at temperature 0, so the same model + same prompt + same
input should give the same answer; storing it lets a re-run (or a crashed run
picked up again) replay instead of paying again. The key covers the model name,
the full system prompt, the full user message and the schema, so any prompt
change is a cache miss, as it should be.

One small JSON file per entry under ``<root>/<2 hex>/<hash>.json``: safe with
many concurrent tasks, and a half-written file is just treated as a miss.
"""

import hashlib
import json
import os
from pathlib import Path


class CachingProvider:
    """Wraps an ``LLMProvider``; same interface, answers from disk when it can."""

    def __init__(self, inner, root: str | Path, chat_model: str, fast_model: str) -> None:
        self.inner = inner
        self.root = Path(root)
        self.models = {True: fast_model, False: chat_model}
        self.embedding_model_id = inner.embedding_model_id
        self.hits = 0
        self.misses = 0
        self.failures = 0  # calls that failed even after the provider's retries

    def _path(self, *parts: str) -> Path:
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def _read(self, path: Path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1
            return None
        self.hits += 1
        return value

    def _write(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{id(value)}.tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        os.replace(tmp, path)  # atomic: readers never see half a file

    async def chat_json(self, system, user, schema, *, fast=True):
        path = self._path("json", self.models[fast], schema.__name__,
                          json.dumps(schema.model_json_schema(), sort_keys=True), system, user)
        cached = self._read(path)
        if cached is not None:
            return schema.model_validate(cached)
        result = await self._call(self.inner.chat_json(system, user, schema, fast=fast))
        self._write(path, result.model_dump(mode="json"))
        return result

    async def chat_text(self, system, user, *, fast=True):
        path = self._path("text", self.models[fast], system, user)
        cached = self._read(path)
        if cached is not None:
            return cached
        result = await self._call(self.inner.chat_text(system, user, fast=fast))
        self._write(path, result)
        return result

    async def embed(self, text):
        path = self._path("embed", self.embedding_model_id, text)
        cached = self._read(path)
        if cached is not None:
            return cached
        result = await self._call(self.inner.embed(text))
        self._write(path, result)
        return result

    async def _call(self, coro):
        try:
            return await coro
        except Exception:
            self.failures += 1
            raise

    async def close(self) -> None:
        await self.inner.close()
