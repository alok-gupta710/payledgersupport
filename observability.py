"""
observability.py: structured logs, per-request latency by stage, error capture and metrics.

Rules:
- Logs are JSON lines (one event per line), rotated by size.
- Logs never contain message text, names, card or bank details: only IDs, lengths, routes, timings, error kinds.
- Every request gets a request_id that appears in the app log, the metrics file and LangSmith metadata.
"""
import contextvars
import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "level": record.levelname,
                   "event": record.getMessage()}
        payload.update(getattr(record, "fields", {}) or {})
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logger(log_dir: Path, level: str = "INFO", max_mb: float = 5.0, backups: int = 3,
                 name: str = "support_app") -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_configured_dir", None) == str(log_dir):
        return logger
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "app.jsonl", maxBytes=int(max_mb * 1024 * 1024), backupCount=backups,
                                 encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    except OSError:
        pass  # no writable log dir: console only (doctor reports it)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    logger._configured_dir = str(log_dir)  # type: ignore[attr-defined]
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


# ---------------------------------------------------------------- request context
@dataclass
class RequestContext:
    request_id: str
    session_id: str = ""
    started: float = field(default_factory=time.perf_counter)
    stages_ms: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: Counter = field(default_factory=Counter)
    errors: List[Dict[str, str]] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)

    def add_error(self, component: str, kind: str, exc_type: str = "") -> None:
        self.errors.append({"component": component, "kind": kind, "exc_type": exc_type})

    def degrade(self, what: str) -> None:
        if what not in self.degraded:
            self.degraded.append(what)

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 1)


_CTX: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar("request_ctx", default=None)


def new_context(session_id: str = "") -> RequestContext:
    ctx = RequestContext(request_id=f"req-{uuid.uuid4().hex[:12]}", session_id=session_id)
    _CTX.set(ctx)
    return ctx


def current() -> Optional[RequestContext]:
    return _CTX.get()


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a block and add it to the current request's stage totals (no-op outside a request)."""
    ctx, t0 = current(), time.perf_counter()
    try:
        yield
    finally:
        if ctx is not None:
            ctx.stages_ms[name] += (time.perf_counter() - t0) * 1000
            ctx.calls[name] += 1


# ---------------------------------------------------------------- metrics
class MetricsRecorder:
    """One JSON line per request: latency by stage, outcome, error kinds, degraded modes, tokens."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, row: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass

    def load(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-last_n:] if last_n else rows


def summarise_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """p50 / p95 / p99 latency overall and per stage, error and degraded rates."""
    import numpy as np

    if not rows:
        return {"requests": 0}
    total = np.array([r["latency_ms"] for r in rows])
    out: Dict[str, Any] = {"requests": len(rows), "p50_ms": round(float(np.percentile(total, 50)), 1),
                           "p95_ms": round(float(np.percentile(total, 95)), 1),
                           "p99_ms": round(float(np.percentile(total, 99)), 1),
                           "error_rate": round(sum(bool(r.get("errors")) for r in rows) / len(rows), 3),
                           "degraded_rate": round(sum(bool(r.get("degraded")) for r in rows) / len(rows), 3),
                           "unhandled_exceptions": sum(r.get("outcome") == "exception" for r in rows)}
    stages = sorted({k for r in rows for k in r.get("stages_ms", {})})
    out["stages"] = {s: {"p50_ms": round(float(np.percentile([r["stages_ms"].get(s, 0) for r in rows], 50)), 1),
                         "p95_ms": round(float(np.percentile([r["stages_ms"].get(s, 0) for r in rows], 95)), 1)}
                     for s in stages}
    out["error_kinds"] = dict(Counter(e["kind"] for r in rows for e in r.get("errors", [])))
    out["degraded_modes"] = dict(Counter(d for r in rows for d in r.get("degraded", [])))
    out["routes"] = dict(Counter(r.get("route") for r in rows))
    return out
