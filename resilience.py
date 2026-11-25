"""
resilience.py: fault injection, circuit breakers, retries, and instrumented wrappers.

Every external dependency (LLM, embeddings, vector store, memory store, feedback store) is wrapped so that:
  - each call is timed into the current request's stage latency,
  - failures are classified and recorded (never the message content),
  - transient LLM errors are retried with exponential backoff and jitter,
  - a circuit breaker stops hammering a failing dependency and lets callers fail fast to a fallback,
  - faults can be injected on purpose to prove the fallbacks work (FAULTS env var or set at runtime).
"""
import logging
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from langchain_core.embeddings import Embeddings

import llm_agent as p3
import memory_agent as p6
import adaptive_agent as p7
import observability as obs


# ---------------------------------------------------------------- injected error types
# Names match the classifier in llm_agent.py, so injected faults are handled exactly like real ones.
class APITimeoutError(TimeoutError):
    pass


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class APIConnectionError(ConnectionError):
    pass


class InjectedError(RuntimeError):
    pass


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a dependency whose circuit is open."""


p3._ERROR_MAP.setdefault("CircuitOpenError", "circuit_open")
p3._ERROR_MAP.setdefault("TimeoutError", "timeout")
p3._ERROR_MAP.setdefault("ConnectionError", "connection_error")
p3._ERROR_MAP.setdefault("OSError", "io_error")
p3._ERROR_MAP.setdefault("InjectedError", "llm_error")
TRANSIENT_KINDS = {"timeout", "rate_limit", "connection_error"}


class FaultPlan:
    """Parse and apply fault injection. Syntax: "component:mode[@n][=seconds], ...".
    modes: timeout | rate_limit | auth | error | down | slow=<seconds>.  '@n' = only the first n calls fail.
    components: llm, embeddings, vector_store, memory_store, feedback_store, search_knowledge_base."""

    MODES = {"timeout", "rate_limit", "auth", "error", "down", "slow"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rules: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def parse(cls, spec: str) -> "FaultPlan":
        plan = cls()
        for part in [p.strip() for p in (spec or "").split(",") if p.strip()]:
            comp, _, rest = part.partition(":")
            plan.set(comp.strip(), rest.strip())
        return plan

    def set(self, component: str, rule: str) -> None:
        if not rule:
            return self.clear(component)
        mode, seconds, remaining = rule, 0.0, None
        if "=" in mode:
            mode, s = mode.split("=", 1)
            seconds = float(s)
        if "@" in mode:
            mode, n = mode.split("@", 1)
            remaining = int(n)
        if mode not in self.MODES:
            raise ValueError(f"Unknown fault mode {mode!r}. Use one of {sorted(self.MODES)}.")
        with self._lock:
            self.rules[component] = {"mode": mode, "seconds": seconds, "remaining": remaining}

    def clear(self, component: Optional[str] = None) -> None:
        with self._lock:
            if component is None:
                self.rules.clear()
            else:
                self.rules.pop(component, None)

    def describe(self) -> str:
        return ", ".join(f"{c}:{r['mode']}" + (f"@{r['remaining']}" if r["remaining"] is not None else "")
                         + (f"={r['seconds']}" if r["seconds"] else "") for c, r in self.rules.items()) or "none"

    def apply(self, component: str, error_family: str = "llm") -> None:
        """Raise (or sleep) according to the rule for this component, if any."""
        with self._lock:
            rule = self.rules.get(component)
            if not rule:
                return
            if rule["remaining"] is not None:
                if rule["remaining"] <= 0:
                    return
                rule["remaining"] -= 1
            mode, seconds = rule["mode"], rule["seconds"]
        if mode == "slow":
            time.sleep(seconds)
            return
        if error_family == "io":
            raise OSError(f"injected {mode} for {component}")
        raise {"timeout": APITimeoutError, "rate_limit": RateLimitError, "auth": AuthenticationError,
               "down": APIConnectionError, "error": InjectedError}[mode](f"injected {mode} for {component}")


# ---------------------------------------------------------------- circuit breaker
class CircuitBreaker:
    """closed -> (N failures within the window) -> open -> (cooldown) -> half_open -> success: closed / failure: open.
    An authentication error opens the circuit at once: retrying a bad key only burns time."""

    def __init__(self, name: str, failures: int = 3, window_s: float = 60, cooldown_s: float = 30,
                 logger: Optional[logging.Logger] = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.name, self.failures, self.window_s, self.cooldown_s = name, failures, window_s, cooldown_s
        self.logger, self.clock = logger, clock
        self.state, self.opened_at = "closed", 0.0
        self._fails: Deque[float] = deque()
        self._lock = threading.Lock()
        self.transitions: List[Tuple[str, str]] = []

    def _move(self, new: str, why: str) -> None:
        if new != self.state:
            self.transitions.append((self.state, new))
            if self.logger:
                obs.log_event(self.logger, "circuit_state_change", logging.WARNING, breaker=self.name,
                              old=self.state, new=new, reason=why)
            self.state = new

    def allow(self) -> bool:
        with self._lock:
            if self.state == "open" and self.clock() - self.opened_at >= self.cooldown_s:
                self._move("half_open", "cooldown elapsed: one trial call allowed")
            return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._fails.clear()
            if self.state == "half_open":
                self._move("closed", "trial call succeeded")

    def record_failure(self, kind: str = "") -> None:
        with self._lock:
            now = self.clock()
            self._fails.append(now)
            while self._fails and now - self._fails[0] > self.window_s:
                self._fails.popleft()
            if self.state == "half_open" or kind == "auth_error" or len(self._fails) >= self.failures:
                self.opened_at = now
                self._move("open", f"{len(self._fails)} failure(s), last kind {kind or 'unknown'}")

    def snapshot(self) -> Dict[str, Any]:
        return {"state": self.state, "recent_failures": len(self._fails)}

    def reset(self) -> None:
        with self._lock:
            self._fails.clear()
            self.state, self.opened_at = "closed", 0.0


# ---------------------------------------------------------------- instrumented LLM
class InstrumentedRunnable:
    """Wraps a bound chat model (or structured-output runnable): faults -> breaker -> timed call -> retry."""

    def __init__(self, inner: Any, stage: str, faults: FaultPlan, breaker: CircuitBreaker, retries: int,
                 base_s: float) -> None:
        self.inner, self.stage, self.faults, self.breaker = inner, stage, faults, breaker
        self.retries, self.base_s = retries, base_s

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kw: Any) -> Any:
        ctx = obs.current()
        for attempt in range(self.retries + 1):
            if not self.breaker.allow():
                if ctx:
                    ctx.add_error(self.stage, "circuit_open", "CircuitOpenError")
                raise CircuitOpenError(f"{self.breaker.name} circuit is open")
            try:
                with obs.stage(self.stage):
                    self.faults.apply("llm")
                    out = self.inner.invoke(input, config=config, **kw)
                self.breaker.record_success()
                return out
            except Exception as exc:  # classify, maybe retry, then re-raise to the agent's own handler
                kind = p3.classify_llm_error(exc)
                if ctx:
                    ctx.add_error(self.stage, kind, type(exc).__name__)
                if kind in TRANSIENT_KINDS and attempt < self.retries:
                    time.sleep(self.base_s * (2 ** attempt) + random.uniform(0, self.base_s / 2))
                    continue
                self.breaker.record_failure(kind)
                raise
        raise RuntimeError("unreachable")


class InstrumentedModel:
    """Stands in for a chat model: bind_tools / with_structured_output return instrumented runnables."""

    def __init__(self, inner: Any, stage: str, faults: FaultPlan, breaker: CircuitBreaker, retries: int = 2,
                 base_s: float = 0.5) -> None:
        self.inner, self.stage, self.faults, self.breaker, self.retries, self.base_s = inner, stage, faults, breaker, retries, base_s

    def _wrap(self, r: Any) -> InstrumentedRunnable:
        return InstrumentedRunnable(r, self.stage, self.faults, self.breaker, self.retries, self.base_s)

    def bind_tools(self, tools: Any, **kw: Any) -> InstrumentedRunnable:
        return self._wrap(self.inner.bind_tools(tools, **kw))

    def with_structured_output(self, schema: Any, **kw: Any) -> InstrumentedRunnable:
        import tool_agent as p5
        if isinstance(self.inner, p5.ScriptedToolModel):
            kw.pop("method", None)
        return self._wrap(self.inner.with_structured_output(schema, **kw))


# ---------------------------------------------------------------- instrumented retrieval dependencies
class InstrumentedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, faults: FaultPlan, breaker: CircuitBreaker, name: str = "embeddings") -> None:
        self.inner, self.faults, self.breaker, self.name = inner, faults, breaker, name

    def _call(self, fn: Callable[[], Any]) -> Any:
        if not self.breaker.allow():
            raise CircuitOpenError(f"{self.breaker.name} circuit is open")
        try:
            with obs.stage("embedding"):
                self.faults.apply(self.name)
                out = fn()
            self.breaker.record_success()
            return out
        except Exception as exc:
            self.breaker.record_failure(p3.classify_llm_error(exc))
            raise

    def embed_query(self, text: str) -> List[float]:
        return self._call(lambda: self.inner.embed_query(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(lambda: self.inner.embed_documents(texts))


class InstrumentedStore:
    def __init__(self, inner: Any, faults: FaultPlan, breaker: CircuitBreaker, name: str = "vector_store") -> None:
        self.inner, self.faults, self.breaker, self.name = inner, faults, breaker, name
        self.backend = inner.backend

    def count(self, namespace: str) -> int:
        return self.inner.count(namespace)

    def upsert(self, *a: Any, **kw: Any) -> None:
        return self.inner.upsert(*a, **kw)

    def wait_until_count(self, *a: Any, **kw: Any) -> None:
        return self.inner.wait_until_count(*a, **kw)

    def query(self, namespace: str, vector: List[float], top_k: int, flt: Optional[Dict[str, Any]] = None):
        if not self.breaker.allow():
            raise CircuitOpenError(f"{self.breaker.name} circuit is open")
        try:
            with obs.stage("vector_search"):
                self.faults.apply(self.name)
                out = self.inner.query(namespace, vector, top_k, flt)
            self.breaker.record_success()
            return out
        except Exception as exc:
            self.breaker.record_failure(p3.classify_llm_error(exc))
            raise


# ---------------------------------------------------------------- stores that degrade instead of crashing
class ResilientMemoryStore(p6.LongTermMemoryStore):
    """Long-term memory that keeps the conversation going if the disk (or DB) fails."""

    def __init__(self, *a: Any, faults: Optional[FaultPlan] = None, logger: Optional[logging.Logger] = None, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.faults, self.logger = faults or FaultPlan(), logger

    def _degraded(self, op: str, exc: Exception) -> None:
        ctx = obs.current()
        if ctx:
            ctx.degrade("memory:unavailable")
            ctx.add_error("memory_store", "io_error", type(exc).__name__)
        if self.logger:
            obs.log_event(self.logger, "memory_store_error", logging.WARNING, op=op, exc_type=type(exc).__name__)

    def load(self, user_key: str) -> Dict[str, Any]:
        try:
            self.faults.apply("memory_store", "io")
            return super().load(user_key)
        except OSError as exc:
            self._degraded("load", exc)
            return self.empty(user_key)

    def save(self, user_key: str, rec: Dict[str, Any]) -> None:
        try:
            self.faults.apply("memory_store", "io")
            super().save(user_key, rec)
        except OSError as exc:
            self._degraded("save", exc)

    def forget(self, user_key: str) -> bool:
        try:
            self.faults.apply("memory_store", "io")
            return super().forget(user_key)
        except OSError as exc:
            self._degraded("forget", exc)
            return False


class ResilientFeedbackStore(p7.FeedbackStore):
    """Feedback that is buffered in memory when storage fails, and flushed when it recovers."""

    def __init__(self, *a: Any, faults: Optional[FaultPlan] = None, logger: Optional[logging.Logger] = None, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.faults, self.logger = faults or FaultPlan(), logger
        self.buffer: List[Dict[str, Any]] = []

    def add(self, raw: Dict[str, Any], tenant: str = p7.TENANT):
        try:
            ev = p7.make_event(raw, tenant)
        except Exception as exc:  # invalid feedback is rejected, not buffered
            self.rejected.append({"record": str({k: raw.get(k) for k in ("source", "kind", "rating", "reason")}),
                                  "error": type(exc).__name__})
            return None
        try:
            self.faults.apply("feedback_store", "io")
            p3.append_jsonl(self.path, ev.model_dump())
        except OSError as exc:
            self.buffer.append(ev.model_dump())
            ctx = obs.current()
            if ctx:
                ctx.degrade("feedback:buffered")
            if self.logger:
                obs.log_event(self.logger, "feedback_store_error", logging.WARNING, exc_type=type(exc).__name__,
                              buffered=len(self.buffer))
        return ev

    def flush(self) -> int:
        written = 0
        while self.buffer:
            try:
                self.faults.apply("feedback_store", "io")
                p3.append_jsonl(self.path, self.buffer[0])
                self.buffer.pop(0)
                written += 1
            except OSError:
                break
        return written
