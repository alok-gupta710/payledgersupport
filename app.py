"""
app.py: Streamlit UI for the PayLedger Support Resolution Agent.

Run:  streamlit run app.py
Every message goes through SupportService.handle(), so the UI gets the same logging, deadlines,
circuit breakers and fallbacks as the tests and the notebook.
"""
import os
import service
import streamlit as st

from service import SupportService

st.set_page_config(page_title="PayLedger Support Assistant", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Starting the support assistant...")
def load_service() -> SupportService:
    return SupportService()


try:
    svc = load_service()
except Exception as exc:  # startup failure: explain, don't show a traceback
    st.error("The assistant could not start. Run `python healthcheck.py` for details.")
    st.caption(f"{type(exc).__name__}")
    st.stop()

REASONS = ["wrong_answer", "too_long", "too_short", "unclear", "should_escalate", "unnecessary_escalation"]

# ---------------- session state ----------------
if "user_id" not in st.session_state:
    st.session_state.user_id = "acme-ap"
if "session_id" not in st.session_state:
    st.session_state.session_id = svc.new_session(st.session_state.user_id)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("PayLedger Support")
    health = svc.health()
    badge = {"ok": "🟢"}.get(health["status"], "🟠")
    st.markdown(f"{badge} **{health['status']}** · v{health['version']}")
    if svc.cfg.dry_run:
        st.warning("DRY RUN: no OpenAI key, answers are placeholders.")
    user = st.text_input("Customer (user ID)", st.session_state.user_id,
                         help="Simulates the signed-in customer. Memory and preferences are per customer.")
    if user != st.session_state.user_id or st.button("New conversation"):
        st.session_state.user_id = user
        st.session_state.session_id = svc.new_session(user)
        st.session_state.messages = []
        st.rerun()
    with st.expander("System health"):
        st.json(health)
    if svc.cfg.environment != "prod":
        with st.expander("Fault injection (testing only)"):
            comp = st.selectbox("Component", ["llm", "embeddings", "vector_store", "search_knowledge_base",
                                              "memory_store", "feedback_store"])
            mode = st.selectbox("Mode", ["timeout", "rate_limit@1", "auth", "error", "down", "slow=5"])
            c1, c2 = st.columns(2)
            if c1.button("Inject"):
                svc.faults.set(comp, mode)
                st.rerun()
            if c2.button("Clear all"):
                svc.faults.clear()
                st.rerun()
            st.caption(f"Active: {svc.faults.describe()}")

chat_tab, ops_tab = st.tabs(["Chat", "Operations"])

# ---------------- chat ----------------
with chat_tab:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                meta = m["meta"]
                st.caption(f"route {meta['route']} · {meta['outcome']} · {meta['latency_ms']:.0f} ms"
                           + (f" · degraded: {', '.join(meta['degraded'])}" if meta["degraded"] else ""))
                with st.expander("Why this answer?"):
                    st.write(meta["explanation"] or "Standard behaviour.")
                    st.json({"request_id": meta["request_id"], "tools": meta["tools"], "stages_ms": meta["stages_ms"],
                             "errors": meta["errors"], "tickets": meta["ticket_ids"]})
                if not m.get("rated"):
                    c1, c2, c3 = st.columns([1, 1, 4])
                    if c1.button("👍", key=f"up{i}"):
                        svc.submit_feedback(st.session_state.session_id, meta["request_id"], "up", "helpful")
                        m["rated"] = True
                        st.rerun()
                    reason = c3.selectbox("Reason", REASONS, key=f"reason{i}", label_visibility="collapsed")
                    if c2.button("👎", key=f"down{i}"):
                        svc.submit_feedback(st.session_state.session_id, meta["request_id"], "down", reason)
                        m["rated"] = True
                        st.rerun()
                else:
                    st.caption("Thanks for the feedback.")

    prompt = st.chat_input("Ask about payments, payouts, fees, invoices or ERP sync")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Checking..."):
            resp = svc.handle(st.session_state.session_id, st.session_state.user_id, prompt)
        st.session_state.messages.append({"role": "assistant", "content": resp.answer, "meta": resp.__dict__})
        st.rerun()

# ---------------- operations ----------------
with ops_tab:
    summary = svc.metrics_summary(last_n=500)
    if summary.get("requests", 0) == 0:
        st.info("No requests yet.")
    else:
        c = st.columns(5)
        c[0].metric("Requests", summary["requests"])
        c[1].metric("p50 latency", f"{summary['p50_ms']:.0f} ms")
        c[2].metric("p95 latency", f"{summary['p95_ms']:.0f} ms")
        c[3].metric("Error rate", f"{summary['error_rate']:.0%}")
        c[4].metric("Degraded rate", f"{summary['degraded_rate']:.0%}")
        st.subheader("Latency by stage (ms)")
        st.dataframe([{"stage": k, **v} for k, v in summary["stages"].items()], width="stretch")
        st.subheader("Errors and degraded modes")
        st.json({"error_kinds": summary["error_kinds"], "degraded_modes": summary["degraded_modes"],
                 "routes": summary["routes"], "unhandled_exceptions": summary["unhandled_exceptions"]})
    st.caption(f"Logs: {svc.cfg.log_dir}/app.jsonl · metrics: {svc.cfg.log_dir}/metrics.jsonl")
