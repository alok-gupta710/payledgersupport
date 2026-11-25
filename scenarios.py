"""
scenarios.py: failure scenarios for the deployed service (used by the notebook and the tests).

All scenarios share ONE running service. Faults are switched on and off at runtime, and the LLM is a
scripted model (search, then answer) so every run is deterministic and costs nothing. Retrieval, the
fallback chain, breakers, deadlines, stores, logging and metrics are the real code.
"""
import time
from typing import Any, Callable, Dict, List

import memory_agent as p6
import tool_agent as p5

GOOD_ANSWER = "GBP payouts cost USD 5.00 per payout plus a 1.0% FX markup."
QUESTION = "What's the fee for GBP payouts?"


def scripted_factory() -> Callable[[], Any]:
    def make():
        return p5.ScriptedToolModel(script=[
            [("search_knowledge_base", {"query": "GBP international payout fee"})],
            [p6._submit("resolved", GOOD_ANSWER, ["gbp_fee"], ["POL-FEES-001"])],
        ])
    return make


SCENARIOS: List[Dict[str, Any]] = [
    {"id": "F0_normal", "faults": "", "what": "No faults: the baseline."},
    {"id": "F1_llm_timeout", "faults": "llm:timeout", "what": "Every LLM call times out (retried twice with backoff)."},
    {"id": "F2_llm_rate_limit_once", "faults": "llm:rate_limit@1", "what": "One rate-limit error, then the API recovers."},
    {"id": "F3_llm_auth_then_fast_fail", "faults": "llm:auth", "what": "Bad API key: the circuit opens at once; the next request fails fast.", "repeat": 2},
    {"id": "F4_vector_store_down", "faults": "vector_store:down", "what": "Vector store (Pinecone) unreachable."},
    {"id": "F5_embeddings_down", "faults": "embeddings:down", "what": "Embedding API unreachable."},
    {"id": "F6_all_retrieval_down", "faults": "search_knowledge_base:error", "what": "Every knowledge backend fails."},
    {"id": "F7_memory_store_down", "faults": "memory_store:down", "what": "Long-term memory storage fails (disk/DB).",
     "message": "We're on the Growth plan. What's our daily payout limit?"},
    {"id": "F8_slow_llm_deadline", "faults": "llm:slow=3", "what": "LLM hangs past the request deadline (1.5 s for this test).", "timeout_s": 1.5},
    {"id": "F9_oversized_input", "faults": "", "what": "A 3,000-character message.", "message": "x " * 1500},
    {"id": "F10_empty_input", "faults": "", "what": "An empty message.", "message": "   "},
    {"id": "F11_unhandled_bug", "faults": "", "what": "A programming error inside the agent (simulated).", "break_agent": True},
]


def run_fault_matrix(svc, scenarios: List[Dict[str, Any]] = SCENARIOS) -> List[Dict[str, Any]]:
    rows = []
    default_timeout = svc.cfg.request_timeout_s
    for sc in scenarios:
        for b in svc.breakers.values():
            b.reset()
        svc.faults.clear()
        for part in [p for p in sc["faults"].split(",") if p]:
            comp, rule = part.split(":", 1)
            svc.faults.set(comp, rule)
        svc.cfg.request_timeout_s = sc.get("timeout_s", default_timeout)
        sid = svc.new_session(f"fault-{sc['id']}")
        if sc.get("break_agent"):
            sess = svc._session(sid, f"fault-{sc['id']}")
            def boom(*a, **k):
                raise KeyError("simulated bug")
            sess.agent.respond = boom
        for i in range(sc.get("repeat", 1)):
            r = svc.handle(sid, f"fault-{sc['id']}", sc.get("message", QUESTION))
            rows.append({"scenario": sc["id"] + (f" (request {i + 1})" if sc.get("repeat", 1) > 1 else ""),
                         "fault": sc["faults"] or "-", "what": sc["what"], "outcome": r.outcome, "route": r.route,
                         "latency_ms": r.latency_ms, "degraded": ", ".join(r.degraded) or "-",
                         "errors": ", ".join(f"{e['component']}:{e['kind']}" for e in r.errors) or "-",
                         "ticket": (r.ticket_ids or ["-"])[0], "answer": r.answer[:140],
                         "llm_breaker": svc.breakers["llm"].state})
    svc.faults.clear()
    svc.cfg.request_timeout_s = default_timeout
    for b in svc.breakers.values():
        b.reset()
    return rows


def breaker_recovery_timeline(svc, cooldown_s: float = 1.0) -> List[Dict[str, Any]]:
    """auth failure -> open -> fast fail -> cooldown -> half-open trial succeeds -> closed."""
    br = svc.breakers["llm"]
    br.reset()
    old = br.cooldown_s
    br.cooldown_s = cooldown_s
    steps = []

    def step(label: str) -> None:
        sid = svc.new_session("breaker-demo")  # fresh session per step, so each gets a fresh scripted model
        r = svc.handle(sid, "breaker-demo", QUESTION)
        steps.append({"step": label, "outcome": r.outcome, "latency_ms": r.latency_ms, "breaker_after": br.state,
                      "answer": r.answer[:90]})

    svc.faults.set("llm", "auth")
    step("1. key revoked (auth error)")
    step("2. next request while open")
    svc.faults.clear("llm")
    step("3. key fixed, still inside cooldown")
    time.sleep(cooldown_s + 0.1)
    step("4. after cooldown: half-open trial")
    step("5. normal traffic")
    br.cooldown_s = old
    return steps


def feedback_buffering(svc) -> Dict[str, Any]:
    sid = svc.new_session("feedback-demo")
    r = svc.handle(sid, "feedback-demo", QUESTION)
    svc.faults.set("feedback_store", "down")
    stored = svc.submit_feedback(sid, r.request_id, "down", "too_long")
    buffered = len(svc.feedback.buffer)
    svc.faults.clear("feedback_store")
    flushed = svc.feedback.flush()
    return {"accepted_while_store_down": stored, "buffered": buffered, "flushed_after_recovery": flushed,
            "left_in_buffer": len(svc.feedback.buffer)}
