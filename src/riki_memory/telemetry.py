"""What happened on each write: so a write path that quietly saves nothing gets noticed.

Every :meth:`RikiMemory.remember` produces one :class:`WriteReport` (memories proposed / saved / dropped, failures, model
calls, tokens, latency). Reports are added to :class:`WriteStats` (running
totals, also in ``health()``) and passed to the host's ``on_write`` callback,
which can forward them to logs or a metrics system.

The report being filled is held in a context variable, so concurrent writes
(separate asyncio tasks) never mix their numbers. A custom ``LLMProvider`` can
report its own usage with :func:`record_model_call` / :func:`record_embed_call`.
"""

import logging
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)

_current: ContextVar["WriteReport | None"] = ContextVar("riki_write_report", default=None)


@dataclass
class WriteReport:
    user_id: str
    mode: str  # what kind of write ("turn")
    source_key: str | None = None
    proposed: int = 0  # memories the model suggested
    saved: int = 0  # memories returned by the write (new or already known)
    dropped_invalid: int = 0  # suggestions rejected by validation
    failures: int = 0  # model/parse/save errors (a session can fail chunk by chunk)
    errors: list[str] = field(default_factory=list)
    model_calls: int = 0
    embed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens the provider served from its prompt cache
    skipped: bool = False  # nothing worth extracting (chitchat): no model call was made
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.failures == 0

    def as_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def _report() -> "WriteReport | None":
    return _current.get()


def record_model_call(prompt_tokens: int = 0, completion_tokens: int = 0, cached_tokens: int = 0) -> None:
    report = _report()
    if report is not None:
        report.model_calls += 1
        report.prompt_tokens += prompt_tokens or 0
        report.completion_tokens += completion_tokens or 0
        report.cached_tokens += cached_tokens or 0


def record_skipped() -> None:
    report = _report()
    if report is not None:
        report.skipped = True


def record_embed_call() -> None:
    report = _report()
    if report is not None:
        report.embed_calls += 1


def record_proposed(count: int) -> None:
    report = _report()
    if report is not None:
        report.proposed += count


def record_dropped(count: int = 1) -> None:
    report = _report()
    if report is not None:
        report.dropped_invalid += count


def record_failure(error: BaseException) -> None:
    report = _report()
    if report is not None:
        report.failures += 1
        report.errors.append(f"{type(error).__name__}: {str(error)[:300]}")


@asynccontextmanager
async def tracking(user_id: str, mode: str, source_key: str | None = None):
    """Collect one write's numbers; yields the report being filled."""
    report = WriteReport(user_id=user_id, mode=mode, source_key=source_key)
    token = _current.set(report)
    started = time.perf_counter()
    try:
        yield report
    finally:
        report.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        _current.reset(token)


@dataclass
class WriteStats:
    """Running totals since this ``RikiMemory`` was created.

    Watch ``failed_writes`` and ``empty_writes``: a write that proposes nothing
    is often correct (small talk), but a run of them with no errors usually
    means the model's answers are not being understood.
    """

    empty_warning_after: int = 25  # consecutive empty writes before a warning; 0 = never
    writes: int = 0
    failed_writes: int = 0
    empty_writes: int = 0
    consecutive_empty: int = 0
    memories_proposed: int = 0
    memories_saved: int = 0
    dropped_invalid: int = 0
    model_calls: int = 0
    embed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    skipped_writes: int = 0
    latency_ms_total: float = 0.0
    last_error: str | None = None

    def add(self, report: WriteReport) -> None:
        self.writes += 1
        self.memories_proposed += report.proposed
        self.memories_saved += report.saved
        self.dropped_invalid += report.dropped_invalid
        self.model_calls += report.model_calls
        self.embed_calls += report.embed_calls
        self.prompt_tokens += report.prompt_tokens
        self.completion_tokens += report.completion_tokens
        self.cached_tokens += report.cached_tokens
        self.latency_ms_total += report.latency_ms
        if not report.ok:
            self.failed_writes += 1
            self.last_error = report.errors[-1] if report.errors else "unknown error"
            self.consecutive_empty = 0
        elif report.skipped:
            self.skipped_writes += 1  # chitchat skipped on purpose; not a sign of a broken model
        elif report.proposed == 0:
            self.empty_writes += 1
            self.consecutive_empty += 1
            if self.empty_warning_after and self.consecutive_empty == self.empty_warning_after:
                log.warning(
                    "%d writes in a row proposed no memories without any error. If the "
                    "conversations had content, the model's replies are probably not being "
                    "understood (check the model and its structured-output support).",
                    self.consecutive_empty,
                )
        else:
            self.consecutive_empty = 0

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("empty_warning_after")
        data["avg_latency_ms"] = round(self.latency_ms_total / self.writes, 1) if self.writes else 0.0
        return data
