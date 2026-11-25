"""
service.py: the deployable Support Resolution service.

SupportService builds every component once at startup (with checks and fallbacks), keeps per-customer
sessions, and handles each message with a request ID, a hard deadline, stage-level latency, error capture,
circuit breakers and graceful fallbacks. The Streamlit app, the CLI and the tests all go through it.

Fallback chain for knowledge search:  primary (OpenAI embeddings + Pinecone)
                                    -> memory (same embeddings, in-memory index)
                                    -> keyword (offline hashing index, no API at all)
If the LLM is unavailable (circuit open), the customer gets an honest message and a ticket, instantly.
"""
import contextvars
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import config as cfgmod
import observability as obs
import resilience as rz
import llm_agent as p3
import rag_agent as p4
import tool_agent as p5
import memory_agent as p6
import adaptive_agent as p7

UNAVAILABLE_TEXT = ("Our assistant is temporarily unavailable, so I couldn't answer this automatically. "
                    "I've logged your question for our support team (ticket {ticket}). You don't need to send it again.")
TIMEOUT_TEXT = ("This is taking longer than it should, so I've stopped rather than keep you waiting. "
                "I've logged it for our support team (ticket {ticket}). You can also try asking again.")
EXCEPTION_TEXT = ("Something went wrong on our side. Nothing was changed on your account. "
                  "I've logged it for our support team (ticket {ticket}).")


# ---------------------------------------------------------------- toolbox and agent with fallbacks
class ResilientToolBox(p7.AdaptiveToolBox):
    """Phase 7 toolbox whose knowledge search walks a fallback chain, each step guarded by its own breaker."""

    def __init__(self, *a: Any, chain: List[Tuple[str, "p4.RagComponents", rz.CircuitBreaker]],
                 fault_plan: Optional[rz.FaultPlan] = None, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.chain = chain
        self.fault_plan = fault_plan or rz.FaultPlan()

    def search_knowledge_base(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        ctx = obs.current()
        last: Optional[Exception] = None
        for mode, comp, breaker in self.chain:
            if not breaker.allow():
                if ctx:
                    ctx.degrade(f"retrieval:{mode}_skipped_circuit_open")
                continue
            orig_comp, orig_hints = self.comp, self.hint_index
            self.comp = comp
            if mode != "primary":
                self.hint_index = None  # hints were learned with the primary embedding model
            try:
                with obs.stage("retrieval"):
                    self.fault_plan.apply("search_knowledge_base", "io")  # test hook: fails every backend
                    result = p7.AdaptiveToolBox.search_knowledge_base(self, query, top_k)
                breaker.record_success()
                if mode != "primary" and ctx:
                    ctx.degrade(f"retrieval:{mode}")
                return {**result, "retrieval_mode": mode}
            except Exception as exc:  # try the next backend
                breaker.record_failure(p3.classify_llm_error(exc))
                if ctx:
                    ctx.add_error(f"retrieval_{mode}", p3.classify_llm_error(exc), type(exc).__name__)
                last = exc
            finally:
                self.comp, self.hint_index = orig_comp, orig_hints
        if ctx:
            ctx.degrade("retrieval:unavailable")
        raise ConnectionError(f"all knowledge backends unavailable ({type(last).__name__ if last else 'circuits open'})")


class ServiceAgent(p7.AdaptiveAgent):
    def __init__(self, *a: Any, chain: List[Tuple[str, "p4.RagComponents", rz.CircuitBreaker]], model_label: str,
                 fault_plan: Optional[rz.FaultPlan] = None, **kw: Any) -> None:
        super().__init__(*a, **kw)
        old = self.box
        self.box = ResilientToolBox(old.tools, old.comp, old.faults, old.session_ref, hint_index=old.hint_index,
                                    chain=chain, fault_plan=fault_plan)
        self.model_name = model_label


# ---------------------------------------------------------------- service
@dataclass
class TurnResponse:
    request_id: str
    answer: str
    route: str
    outcome: str                 # ok | degraded | fallback | timeout | exception
    latency_ms: float
    stages_ms: Dict[str, float]
    degraded: List[str]
    errors: List[Dict[str, str]]
    ticket_ids: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    adaptations: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class Session:
    session_id: str
    user_id: str
    agent: Optional[ServiceAgent]
    last_active: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    turns: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # request_id -> facts needed for feedback


class SupportService:
    def __init__(self, cfg: Optional[cfgmod.AppConfig] = None,
                 model_factory: Optional[Callable[[], Any]] = None) -> None:
        t0 = time.perf_counter()
        self.cfg = cfg or cfgmod.load_config()
        self.logger = obs.setup_logger(self.cfg.log_dir, self.cfg.log_level, self.cfg.log_max_mb, self.cfg.log_backups)
        self.metrics = obs.MetricsRecorder(self.cfg.log_dir / "metrics.jsonl")
        self.faults = rz.FaultPlan.parse(self.cfg.faults)
        self.settings = p3.load_settings()
        self.cfg.dry_run = self.settings.dry_run
        self.model_factory = model_factory
        self.started_at = datetime.now(timezone.utc)
        mk = lambda name: rz.CircuitBreaker(name, self.cfg.breaker_failures, self.cfg.breaker_window_s,
                                            self.cfg.breaker_cooldown_s, self.logger)
        self.breakers = {n: mk(n) for n in ["llm", "embeddings", "vector_store", "retrieval_primary",
                                            "retrieval_memory", "retrieval_keyword"]}
        self.startup_warnings: List[str] = []
        self._build_knowledge()
        self.prompt = p6.load_p6_prompt()
        self.memory_store = rz.ResilientMemoryStore(root=self.cfg.memory_dir, faults=self.faults, logger=self.logger)
        self.feedback = rz.ResilientFeedbackStore(self.cfg.feedback_path, faults=self.faults, logger=self.logger)
        self.registry = self._load_registry()
        self.hint_index = self._build_hints()
        self.sessions: Dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="turn")
        self.manifest = {
            "app_version": self.cfg.app_version, "environment": self.cfg.environment,
            "llm": "dry-run (scripted)" if self.settings.dry_run else self.settings.model,
            "embeddings": self.primary.embedding_name, "vector_store": self.primary.store.backend,
            "fallbacks": [m for m, _, _ in self.chain[1:]], "threshold": self.primary.threshold,
            "corpus_fingerprint": self.fingerprint, "prompt": f"{self.prompt.id} ({self.prompt.sha})",
            "registry_version": self.registry.version, "active_adaptations": len(self.registry.active()),
            "features": {"planning": self.cfg.planning, "memory": self.cfg.memory, "adaptation": self.cfg.adaptation},
            "langsmith_tracing": self.settings.langsmith_tracing, "faults": self.faults.describe(),
            "startup_warnings": self.startup_warnings, "startup_s": round(time.perf_counter() - t0, 2)}
        obs.log_event(self.logger, "service_started", **self.manifest)

    # ---------- startup ----------
    def _build_knowledge(self) -> None:
        rs = p4.load_rag_settings(self.settings)
        docs = p4.load_corpus()
        self.chunks = p4.build_chunks(docs, rs.chunking if rs.chunking in p4.CHUNKERS else "section")
        self.fingerprint = p4.corpus_fingerprint(docs)
        self.kb_texts = {d.doc_id: d.body for d in docs if d.status == "active"}
        active = [c for c in self.chunks if c.status == "active"]
        db = p3.load_mock_db()
        cases = p4.load_rag_cases(p4.CASES_PATH)
        tools = p3.MockTools(db, "T-001")
        tokens = {c.chunk_id: c.tokens for c in self.chunks}

        def comp(name: str, emb: Any, store: Any, ns: str, thr: float) -> p4.RagComponents:
            return p4.RagComponents(self.settings, rs, emb, name, store, ns, thr, active, p4.load_p4_prompts(),
                                    p4.load_rag_fewshot(), db)

        def calibrate(emb: Any, store: Any, ns: str) -> float:
            if rs.threshold_override is not None:
                return rs.threshold_override
            pq = p4.evaluate_retrieval(cases, emb, store, ns, tools, tokens)
            return p4.calibrate_threshold(pq)[0]

        # keyword fallback: always available, no network
        kw_emb, kw_store = p4.HashingEmbeddings(), p4.InMemoryVectorStore()
        kw_ns = p4.namespace_for("hashing-lexical", "section", self.fingerprint)
        p4.build_index(self.chunks, kw_emb, kw_store, kw_ns)
        keyword = comp("hashing-lexical", kw_emb, kw_store, kw_ns, calibrate(kw_emb, kw_store, kw_ns))

        chain: List[Tuple[str, p4.RagComponents, rz.CircuitBreaker]] = []
        name = rs.embedding_model
        try:
            raw_emb, dim = p4.make_embeddings(name, self.settings) if name != "hashing-lexical" else (p4.HashingEmbeddings(), 1024)
            emb = rz.InstrumentedEmbeddings(raw_emb, self.faults, self.breakers["embeddings"])
            raw_store = p4.make_store(rs, dim) if name != "hashing-lexical" else p4.InMemoryVectorStore()
            store = rz.InstrumentedStore(raw_store, self.faults, self.breakers["vector_store"])
            ns = p4.namespace_for(name, "section", self.fingerprint)
            p4.build_index(self.chunks, emb, store, ns)
            thr = self._cached_threshold(name, lambda: calibrate(emb, store, ns))
            primary = comp(name, emb, store, ns, thr)
            chain.append(("primary", primary, self.breakers["retrieval_primary"]))
            if store.backend == "pinecone":  # second line: same embeddings, local index
                mem_store = p4.InMemoryVectorStore()
                p4.build_index(self.chunks, emb, mem_store, ns)
                chain.append(("memory", comp(name, emb, mem_store, ns, thr), self.breakers["retrieval_memory"]))
        except Exception as exc:  # primary retrieval unavailable at startup: start degraded, not dead
            self.startup_warnings.append(f"primary retrieval unavailable at startup ({type(exc).__name__}); using keyword search")
            obs.log_event(self.logger, "startup_degraded", logging.ERROR, component="retrieval", exc_type=type(exc).__name__)
            primary = keyword
        chain.append(("keyword", keyword, self.breakers["retrieval_keyword"]))
        self.primary, self.keyword, self.chain = primary, keyword, chain

    def _cached_threshold(self, emb_name: str, compute: Callable[[], float]) -> float:
        path = self.cfg.log_dir.parent / "results" / "retrieval_config.json"
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if cfg.get("embedding") == emb_name and cfg.get("corpus_fingerprint") == self.fingerprint:
                return float(cfg["threshold"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass
        thr = compute()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"embedding": emb_name, "chunking": "section", "threshold": thr,
                                        "corpus_fingerprint": self.fingerprint}, indent=2), encoding="utf-8")
        except OSError:
            pass
        return thr

    def _load_registry(self) -> p7.AdaptationRegistry:
        try:
            return p7.AdaptationRegistry(self.cfg.registry_path)
        except Exception as exc:  # corrupt registry: run without adaptations
            self.startup_warnings.append(f"adaptation registry unreadable ({type(exc).__name__}); adaptations off")
            return p7.AdaptationRegistry(self.cfg.registry_path.with_name("_unreadable_fallback.json"))

    def _build_hints(self) -> Optional[p7.HintIndex]:
        if not self.cfg.adaptation or not self.registry.active("retrieval_hint"):
            return None
        try:
            idx = p7.HintIndex(self.primary.embeddings, self.chunks, float(self.registry_hint_threshold()))
            for a in self.registry.active("retrieval_hint"):
                idx.add(a.id, a.change["query"], a.change["doc_id"])
            return idx
        except Exception as exc:
            self.startup_warnings.append(f"retrieval hints not loaded ({type(exc).__name__})")
            return None

    def registry_hint_threshold(self) -> float:
        """Use the threshold chosen by the Phase 7 sweep (HINT_THRESHOLD), else a conservative default."""
        import os
        raw = os.getenv("HINT_THRESHOLD", "").strip()
        if raw:
            return float(raw)
        return {"hashing-lexical": 0.35}.get(self.primary.embedding_name, 0.8)

    # ---------- sessions ----------
    def _new_agent(self, user_id: str) -> ServiceAgent:
        base = self.model_factory() if self.model_factory else p5.make_tool_model(self.settings)
        chat = rz.InstrumentedModel(base, "llm", self.faults, self.breakers["llm"], self.cfg.llm_retries, self.cfg.retry_base_s)
        planner = None
        if self.cfg.planning and not self.settings.dry_run and self.model_factory is None:
            planner = rz.InstrumentedModel(p3.make_chat_model(self.settings), "planner", self.faults, self.breakers["llm"],
                                           self.cfg.llm_retries, self.cfg.retry_base_s)
        agent = ServiceAgent(self.primary, self.prompt, chat, registry=self.registry, hint_index=self.hint_index,
                             feedback_store=self.feedback, adapt=self.cfg.adaptation, planner_model=planner,
                             store=self.memory_store, user_id=user_id, planning=self.cfg.planning, memory=self.cfg.memory,
                             chain=self.chain, fault_plan=self.faults, model_label="scripted" if (self.settings.dry_run or self.model_factory) else self.settings.model,
                             log_path=self.cfg.log_dir / "agent_turns.jsonl")
        agent.log_path = self.cfg.log_dir / "agent_turns.jsonl"
        return agent

    def new_session(self, user_id: str) -> str:
        sid = f"s-{uuid.uuid4().hex[:10]}"
        with self._sessions_lock:
            self._evict()
            self.sessions[sid] = Session(sid, user_id, None, time.monotonic())
        return sid

    def _evict(self) -> None:
        now = time.monotonic()
        for sid in [s for s, v in self.sessions.items() if now - v.last_active > self.cfg.session_idle_s]:
            self.sessions.pop(sid, None)
        while len(self.sessions) >= self.cfg.max_sessions:
            oldest = min(self.sessions.values(), key=lambda s: s.last_active)
            self.sessions.pop(oldest.session_id, None)

    def _session(self, session_id: str, user_id: str) -> Session:
        with self._sessions_lock:
            s = self.sessions.get(session_id)
            if s is None:
                self._evict()
                s = self.sessions[session_id] = Session(session_id, user_id, None, time.monotonic())
        if s.agent is None:
            s.agent = self._new_agent(user_id)
        s.last_active = time.monotonic()
        return s

    # ---------- one message ----------
    def handle(self, session_id: str, user_id: str, message: Any) -> TurnResponse:
        ctx = obs.new_context(session_id)
        user_hash = p6.LongTermMemoryStore.user_key("T-001", user_id)[:10]
        obs.log_event(self.logger, "request_start", request_id=ctx.request_id, session_id=session_id, user=user_hash,
                      message_chars=len(message) if isinstance(message, str) else -1)
        answer, route, outcome, tickets, tools, adaptations, explanation = "", "escalated", "ok", [], [], [], ""
        sess: Optional[Session] = None
        try:
            sess = self._session(session_id, user_id)
            if not self.breakers["llm"].allow():
                ctx.degrade("llm:circuit_open")
                ticket = sess.agent.box.tools.create_ticket("general", "normal", "Assistant unavailable (LLM circuit open).")
                answer, outcome, tickets = UNAVAILABLE_TEXT.format(ticket=ticket), "fallback", [ticket]
            else:
                with sess.lock:
                    run = contextvars.copy_context().run
                    fut = self.executor.submit(run, sess.agent.respond, message,
                                               {"request_id": ctx.request_id, "app_version": self.cfg.app_version})
                    try:
                        r = fut.result(timeout=self.cfg.request_timeout_s)
                    except FuturesTimeout:
                        ctx.add_error("request", "deadline_exceeded", "TimeoutError")
                        sess.agent = None  # the late thread must not keep mutating this session
                        fresh = self._new_agent(user_id)
                        ticket = fresh.box.tools.create_ticket("general", "normal", "Request exceeded the deadline.")
                        sess.agent = fresh
                        answer, outcome, tickets = TIMEOUT_TEXT.format(ticket=ticket), "timeout", [ticket]
                    else:
                        answer, route, tickets, tools = r.answer, r.route, r.ticket_ids, r.tools_attempted
                        adaptations = r.adaptations_applied
                        explanation = p7.explain_turn(r, self.registry)
                        if r.error:
                            outcome = "fallback"
                        elif ctx.degraded:
                            outcome = "degraded"
                        sess.turns[ctx.request_id] = {"message": p3.redact_pii(message)[0] if isinstance(message, str) else "",
                                                      "adaptations": adaptations, "route": route}
        except Exception as exc:  # a bug: never show a stack trace, never lose the request
            tb = traceback.extract_tb(exc.__traceback__)
            where = f"{tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}" if tb else "?"
            ctx.add_error("service", "unhandled_exception", type(exc).__name__)
            obs.log_event(self.logger, "unhandled_exception", logging.ERROR, request_id=ctx.request_id,
                          exc_type=type(exc).__name__, where=where)
            ticket = f"TCK-{uuid.uuid4().hex[:6].upper()}"
            answer, outcome, tickets = EXCEPTION_TEXT.format(ticket=ticket), "exception", [ticket]
            if sess is not None:
                sess.agent = None
        latency = ctx.elapsed_ms()
        resp = TurnResponse(ctx.request_id, answer, route, outcome, latency, {k: round(v, 1) for k, v in ctx.stages_ms.items()},
                            list(ctx.degraded), list(ctx.errors), tickets, tools, adaptations, explanation)
        self.metrics.record({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": ctx.request_id,
                             "session_id": session_id, "outcome": outcome, "route": route, "latency_ms": latency,
                             "stages_ms": resp.stages_ms, "errors": resp.errors, "degraded": resp.degraded,
                             "tools": tools, "adaptations": adaptations, "faults": self.faults.describe()})
        obs.log_event(self.logger, "request_end", logging.WARNING if outcome in {"timeout", "exception"} else logging.INFO,
                      request_id=ctx.request_id, outcome=outcome, route=route, latency_ms=latency,
                      stages_ms=resp.stages_ms, errors=resp.errors, degraded=resp.degraded, tools=tools)
        return resp

    # ---------- feedback, health, metrics ----------
    def submit_feedback(self, session_id: str, request_id: str, rating: str, reason: Optional[str] = None) -> bool:
        s = self.sessions.get(session_id)
        turn = s.turns.get(request_id) if s else None
        if not turn:
            return False
        ev = self.feedback.add({"source": "customer", "kind": "rating", "user_id": s.user_id, "message": turn["message"],
                                "rating": rating, "reason": reason, "adaptations_active": turn["adaptations"]})
        obs.log_event(self.logger, "feedback", request_id=request_id, rating=rating, reason=reason, stored=ev is not None,
                      buffered=len(self.feedback.buffer))
        return ev is not None

    def health(self) -> Dict[str, Any]:
        mem_ok = cfgmod._writable(self.cfg.memory_dir)
        status = "ok"
        if any(b.state != "closed" for b in self.breakers.values()) or self.startup_warnings or self.feedback.buffer or not mem_ok:
            status = "degraded"
        if self.breakers["llm"].state == "open" and not self.settings.dry_run:
            status = "degraded (LLM unavailable)"
        return {"status": status, "version": self.cfg.app_version, "uptime_s": int((datetime.now(timezone.utc) - self.started_at).total_seconds()),
                "llm": self.manifest["llm"], "retrieval_chain": [m for m, _, _ in self.chain],
                "breakers": {k: v.snapshot() for k, v in self.breakers.items()},
                "memory_store_writable": mem_ok, "feedback_buffered": len(self.feedback.buffer),
                "sessions": len(self.sessions), "registry_version": self.registry.version,
                "active_adaptations": len(self.registry.active()), "faults": self.faults.describe(),
                "startup_warnings": self.startup_warnings}

    def metrics_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        return obs.summarise_metrics(self.metrics.load(last_n))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        obs.log_event(self.logger, "service_stopped")


_SINGLETON: Optional[SupportService] = None
_SINGLETON_LOCK = threading.Lock()


def get_service() -> SupportService:
    """Process-wide service (the Streamlit app also caches it with st.cache_resource)."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = SupportService()
        return _SINGLETON
