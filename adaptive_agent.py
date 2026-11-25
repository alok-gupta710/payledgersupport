"""
adaptive_agent.py
Phase 7: adaptive behaviour from feedback for the Support Resolution Agent.

Feedback (explicit ratings, specialist reviews, implicit conversation signals)
  -> validated, PII-redacted FeedbackStore (JSONL, 180-day retention)
  -> AdaptationEngine: four bounded adaptation types, each with evidence thresholds and a gate
       1. reply_style       per customer   (repeated "too long" -> brief, enforced in code)
       2. retrieval_hint    global         (specialist corrections; accepted only if a retrieval gate shows no harm)
       3. intent_guidance   per topic      (ticket/answer reviews; relaxing needs approval and is blocked for safety topics)
       4. learned_example   per topic      (specialist-approved answers, verified against the knowledge base)
  -> versioned AdaptationRegistry (audit history, rollback) and post-change monitoring (auto-disable)
  -> AdaptiveAgent applies active adaptations and records which ones shaped each answer.

Reuses Phases 3-6. Python 3.10 (Vocareum). Keys in .env.
"""

# %% [imports]
import hashlib
import json
import re
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError, field_validator

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import llm_agent as p3
import rag_agent as p4
import tool_agent as p5
import memory_agent as p6
from llm_agent import DataLoadError, SetupError

EVAL_DIR = BASE_DIR / "eval"
FEEDBACK_DIR = BASE_DIR / "feedback"
ADAPT_DIR = BASE_DIR / "adaptations"
RESULTS_DIR = BASE_DIR / "results"
ADAPT_LOG = BASE_DIR / "logs" / "adaptive_interactions.jsonl"
FEEDBACK_RETENTION_DAYS = 180
TENANT = "T-001"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# %% [taxonomy]
# Deterministic topic labels, used to group feedback and to decide which learned guidance applies to a message.
INTENT_PATTERNS: Dict[str, str] = {
    "erp_rate_limit": r"\bERP-?429\b|too many requests|rate[- ]?limit",
    "erp_error": r"\bERP-?\d{3}\b|\bsync (failed|error)",
    "unauthorized_return": r"\bR(10|29)\b|unauthori[sz]ed (debit|return)",
    "chargeback_dispute": r"chargeback|dispute",
    "fraud_takeover": r"fraud|hack|takeover|unauthori[sz]ed (payout|access|login|payments?)",
    "bank_change": r"\b(change|switch|update|deposit|route|send)\b.{0,40}\b(bank|account|payouts? to)\b",
    "payment_cancel": r"\bcancel|\bstop a pay|\brecall",
    "approvals": r"approv|sign[- ]off",
    "user_access": r"\b(add|invite|remove|give|set up)\b.{0,30}\b(user|colleague|coworker|access)\b|\brole\b|viewer|read-only|look-only|look-but",
    "limits": r"\blimit\b",
    "fees": r"\bfees?\b|\bcost\b|pricing",
    "settlement": r"how long|settle|arrive|\bland\b|cut-?off",
    "refund_duplicate": r"refund|charged twice|duplicate",
    "payment_status": r"\bPAY-?\d+|\bstatus\b",
}
INTENT_DESC = {
    "erp_rate_limit": "ERP-429 (ERP rate limit) sync errors", "unauthorized_return": "R10 or R29 unauthorized returns",
    "chargeback_dispute": "chargebacks and disputes", "payment_cancel": "cancelling payments",
    "limits": "payout limits", "fees": "fees",
}
# Topics where an adaptation may make the agent MORE careful but never less.
SAFETY_LOCKED = {"unauthorized_return", "chargeback_dispute", "fraud_takeover", "bank_change"}


def classify_intents(text: str) -> List[str]:
    labels = [k for k, pat in INTENT_PATTERNS.items() if re.search(pat, text, re.IGNORECASE)]
    return labels or ["other"]


# %% [feedback]
REASONS = Literal["too_long", "too_short", "wrong_answer", "unclear", "should_escalate", "unnecessary_escalation",
                  "didnt_use_my_info", "rephrase", "complaint", "asked_for_human", "helpful"]
NEGATIVE_REASONS = {"wrong_answer", "unclear", "should_escalate", "unnecessary_escalation", "didnt_use_my_info",
                    "rephrase", "complaint", "asked_for_human"}


class FeedbackEvent(BaseModel):
    """One feedback signal. Free text is redacted and capped; the customer is a hash, never a name."""
    feedback_id: str = Field(default_factory=lambda: f"FB-{uuid.uuid4().hex[:8]}")
    ts: str = Field(default_factory=lambda: utcnow().isoformat(timespec="seconds"))
    source: Literal["customer", "specialist", "implicit"]
    kind: Literal["rating", "answer_review", "ticket_review", "retrieval_correction", "approved_example"]
    user_key: Optional[str] = None
    intent: Optional[str] = None
    message: Optional[str] = Field(default=None, max_length=500)
    rating: Optional[Literal["up", "down"]] = None
    reason: Optional[REASONS] = None
    verdict: Optional[Literal["correct_escalation", "could_self_serve", "should_have_escalated"]] = None
    expected_team: Optional[p3.Team] = None
    correct_doc_id: Optional[str] = None
    approved_answer: Optional[str] = Field(default=None, max_length=800)
    approved_citations: List[str] = Field(default_factory=list)
    approved_route: Optional[Literal["resolved", "needs_info", "escalated", "refused"]] = None
    approved_by: Optional[str] = None
    comment: Optional[str] = Field(default=None, max_length=500)
    adaptations_active: List[str] = Field(default_factory=list)

    @field_validator("message", "comment", "approved_answer")
    @classmethod
    def _redact(cls, v: Optional[str]) -> Optional[str]:
        return p3.redact_pii(v)[0] if v else v


def style_adaptation_id(user_key: str) -> str:
    return f"A-style-{user_key[:8]}"


def resolve_adaptation_ref(ref: str, tenant: str = TENANT) -> str:
    """Readable references like 'A-style-acme-ap' (used in YAML files) -> the hashed per-customer ID."""
    m = re.match(r"^A-style-(?![0-9a-f]{8}$)([\w-]+)$", ref)
    return style_adaptation_id(p6.LongTermMemoryStore.user_key(tenant, m.group(1))) if m else ref


def make_event(raw: Dict[str, Any], tenant: str = TENANT) -> FeedbackEvent:
    """Turn a raw record (e.g. from the UI or a YAML file) into a validated event."""
    data = dict(raw)
    uid = data.pop("user_id", None)
    if uid:
        data["user_key"] = p6.LongTermMemoryStore.user_key(tenant, uid)
    data["adaptations_active"] = [resolve_adaptation_ref(r, tenant) for r in data.get("adaptations_active", [])]
    ev = FeedbackEvent(**data)
    if not ev.intent and ev.message:
        ev.intent = classify_intents(ev.message)[0]
    return ev


class FeedbackStore:
    """Append-only JSONL. Invalid events are rejected with a reason, never half-stored."""

    def __init__(self, path: Path = FEEDBACK_DIR / "feedback.jsonl") -> None:
        self.path = Path(path)
        self.rejected: List[Dict[str, str]] = []

    def add(self, raw: Dict[str, Any], tenant: str = TENANT) -> Optional[FeedbackEvent]:
        try:
            ev = make_event(raw, tenant)
        except ValidationError as exc:
            self.rejected.append({"record": str({k: raw.get(k) for k in ("source", "kind", "rating", "reason")}),
                                  "error": p5._validation_message(exc)})
            return None
        p3.append_jsonl(self.path, ev.model_dump())
        return ev

    def add_many(self, raws: List[Dict[str, Any]]) -> List[FeedbackEvent]:
        return [e for e in (self.add(r) for r in raws) if e is not None]

    def load(self, since: Optional[datetime] = None) -> List[FeedbackEvent]:
        events = [FeedbackEvent(**r) for r in p3.read_jsonl(self.path)]
        return [e for e in events if since is None or datetime.fromisoformat(e.ts) >= since]

    def prune(self, now: datetime) -> int:
        """Retention: drop events older than FEEDBACK_RETENTION_DAYS. Returns how many were removed."""
        keep = [e for e in self.load() if datetime.fromisoformat(e.ts) >= now - timedelta(days=FEEDBACK_RETENTION_DAYS)]
        removed = len(p3.read_jsonl(self.path)) - len(keep)
        if removed:
            self.path.write_text("".join(json.dumps(e.model_dump()) + "\n" for e in keep), encoding="utf-8")
        return removed


_COMPLAINT = re.compile(r"\b(that'?s (wrong|not right|incorrect)|not what i asked|not helpful|useless|you already asked|"
                        r"doesn'?t answer|didn'?t answer)\b", re.IGNORECASE)
_THANKS = re.compile(r"\b(thanks|thank you|perfect|that worked|great,? (thanks|that)|solved)\b", re.IGNORECASE)
_HUMAN = re.compile(r"\b(human|real person|talk to (someone|an agent)|representative)\b", re.IGNORECASE)


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(p4._tokens(a)), set(p4._tokens(b))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def detect_implicit(prev: Optional[Dict[str, Any]], current_redacted: str) -> Optional[Dict[str, Any]]:
    """Signals about the PREVIOUS answer, read from how the customer reacts to it."""
    if not prev:
        return None
    reason = None
    if _COMPLAINT.search(current_redacted):
        reason = "complaint"
    elif prev["route"] == "resolved" and _HUMAN.search(current_redacted):
        reason = "asked_for_human"
    elif prev["route"] in {"resolved", "needs_info"} and _jaccard(prev["message"], current_redacted) >= 0.5:
        reason = "rephrase"
    elif _THANKS.search(current_redacted):
        return {"rating": "up", "reason": "helpful"}
    return {"rating": "down", "reason": reason} if reason else None


# %% [adaptations]
@dataclass
class Adaptation:
    id: str
    kind: str              # reply_style | retrieval_hint | intent_guidance | learned_example
    scope: str             # customer:<key> | global | intent:<label>
    change: Dict[str, Any]
    reason: str            # plain-language "why"
    evidence: Dict[str, Any]
    risk: str              # low | medium
    status: str = "proposed"   # proposed | active | rejected | pending_approval | disabled | rolled_back
    gate: Dict[str, Any] = field(default_factory=dict)
    approved_by: Optional[str] = None
    history: List[str] = field(default_factory=list)


class AdaptationRegistry:
    """Versioned list of adaptations with an audit trail. Every status change bumps the version."""

    def __init__(self, path: Path = ADAPT_DIR / "registry.json") -> None:
        self.path = Path(path)
        self.version = 0
        self.items: Dict[str, Adaptation] = {}
        self.audit: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.version, self.audit = raw["version"], raw["audit"]
                self.items = {a["id"]: Adaptation(**a) for a in raw["items"]}
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise DataLoadError(f"Adaptation registry {self.path.name} is unreadable ({type(exc).__name__}).") from exc

    def _log(self, adaptation_id: str, action: str, by: str, note: str) -> None:
        self.version += 1
        entry = {"version": self.version, "ts": utcnow().isoformat(timespec="seconds"), "adaptation": adaptation_id,
                 "action": action, "by": by, "note": note}
        self.audit.append(entry)
        if adaptation_id in self.items:
            self.items[adaptation_id].history.append(f"v{self.version} {action} by {by}: {note}")

    def upsert(self, a: Adaptation, by: str = "engine") -> None:
        old = self.items.get(a.id)
        if old and old.status in {"active", "disabled", "rolled_back"} and old.change == a.change:
            return  # already decided; don't flip-flop on re-learning
        self.items[a.id] = a
        self._log(a.id, a.status, by, a.reason if a.status != "rejected" else a.gate.get("reason", a.reason))

    def set_status(self, adaptation_id: str, status: str, by: str, note: str) -> Adaptation:
        a = self.items[adaptation_id]
        a.status = status
        if status == "active" and by != "engine":
            a.approved_by = by
        self._log(adaptation_id, status, by, note)
        return a

    def active(self, kind: Optional[str] = None) -> List[Adaptation]:
        return [a for a in self.items.values() if a.status == "active" and (kind is None or a.kind == kind)]

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": self.version, "audit": self.audit,
                                   "items": [asdict(a) for a in self.items.values()]}, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path


# %% [retrieval_hints]
class HintIndex:
    """Specialist-corrected queries -> the document that answered them. A new query close enough to a
    corrected one gets that document's best chunk added to its search results ("feedback_hint")."""

    def __init__(self, embeddings: Any, chunks: List[p4.Chunk], threshold: float) -> None:
        self.embeddings = embeddings
        self.threshold = threshold
        self.by_doc: Dict[str, List[p4.Chunk]] = defaultdict(list)
        for c in chunks:
            if c.status == "active":
                self.by_doc[c.doc_id].append(c)
        self.hints: List[Dict[str, Any]] = []
        self._chunk_vecs: Dict[str, np.ndarray] = {}

    @staticmethod
    def _unit(v: Any) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n else v

    def add(self, hint_id: str, query: str, doc_id: str) -> None:
        if doc_id not in self.by_doc:
            raise ValueError(f"unknown or inactive doc {doc_id}")
        self.hints.append({"id": hint_id, "query": query, "doc_id": doc_id,
                           "vec": self._unit(self.embeddings.embed_query(query))})

    def matches(self, query_vec: Any) -> List[Tuple[Dict[str, Any], float]]:
        q = self._unit(query_vec)
        out = [(h, float(q @ h["vec"])) for h in self.hints]
        return sorted([m for m in out if m[1] >= self.threshold], key=lambda m: -m[1])

    def best_chunk(self, doc_id: str, query_vec: Any) -> p4.Chunk:
        q = self._unit(query_vec)
        best, best_s = None, -2.0
        for c in self.by_doc[doc_id]:
            if c.chunk_id not in self._chunk_vecs:
                self._chunk_vecs[c.chunk_id] = self._unit(self.embeddings.embed_query(c.embed_text))
            s = float(q @ self._chunk_vecs[c.chunk_id])
            if s > best_s:
                best, best_s = c, s
        return best

    def augment(self, query: str, docs_top: List[str]) -> Tuple[List[str], List[str]]:
        """Return (docs after hints, hint ids that fired). Hinted docs not already present are put first."""
        qv = self.embeddings.embed_query(query)
        fired, added = [], []
        for h, _ in self.matches(qv):
            if h["doc_id"] not in docs_top and h["doc_id"] not in added:
                added.append(h["doc_id"])
            fired.append(h["id"])
        return added + docs_top, list(dict.fromkeys(fired))


def hint_gate(index_factory, candidate: Dict[str, str], accepted: List[Dict[str, str]], queries: List[Dict[str, Any]],
              comp: "p4.RagComponents", k: int = 4) -> Dict[str, Any]:
    """Evaluate one candidate hint on top of already-accepted hints.
    Accept only if: it fixes its own query, makes no controlled query worse, and fires on no unanswerable query."""
    def run(hints: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        idx = index_factory()
        for h in hints:
            idx.add(h["id"], h["query"], h["doc_id"])
        out = {}
        for q in queries:
            base = comp.store.query(comp.namespace, comp.embeddings.embed_query(q["query"]), k, {"status": "active"})
            docs, fired = idx.augment(q["query"], [h_.metadata["doc_id"] for h_ in base])
            rel = set(q["relevant_docs"])
            rank = next((i + 1 for i, d in enumerate(docs) if d in rel), None)
            out[q["query"]] = {"rank": rank, "fired": fired, "group": q["group"], "hit": bool(rank and rank <= k + len(fired))}
        return out

    before, after = run(accepted), run(accepted + [candidate])
    needed = not before.get(candidate["query"], {}).get("hit", False)
    own = after.get(candidate["query"], {})
    regressions = [q for q, b in before.items() if b["group"] in {"control", "paraphrase", "training"} and b["hit"] and not after[q]["hit"]]
    false_fires = [q for q, a in after.items() if a["group"] == "unanswerable" and candidate["id"] in a["fired"]]
    improved = [q for q, b in before.items() if b["group"] in {"paraphrase", "training"} and not b["hit"] and after[q]["hit"]]
    ok = needed and bool(own.get("hit")) and not regressions and not false_fires
    return {"accepted": ok, "needed": needed, "fixes_own_query": bool(own.get("hit")), "improved": improved,
            "regressions": regressions, "false_fires": false_fires}


def sweep_hint_threshold(comp: "p4.RagComponents", chunks: List[p4.Chunk], training: List[Dict[str, str]],
                         queries: List[Dict[str, Any]], taus: List[float], k: int = 4) -> List[Dict[str, Any]]:
    """Try each similarity threshold with ALL corrected queries as hints (no gate) and count what happens.
    Pick the lowest threshold with zero false fires and zero regressions that still helps paraphrases."""
    rows = []
    for tau in taus:
        idx = HintIndex(comp.embeddings, chunks, tau)
        for i, t in enumerate(training):
            idx.add(f"h{i}", t["query"], t["doc_id"])
        helped = false_fires = regressions = 0
        for q in queries:
            base = comp.store.query(comp.namespace, comp.embeddings.embed_query(q["query"]), k, {"status": "active"})
            base_docs = [h.metadata["doc_id"] for h in base]
            docs, fired = idx.augment(q["query"], base_docs)
            rel = set(q["relevant_docs"])
            hit_before = bool(rel & set(base_docs))
            hit_after = bool(rel & set(docs))
            if q["group"] == "unanswerable" and fired:
                false_fires += 1
            if q["group"] == "paraphrase" and hit_after and not hit_before:
                helped += 1
            if hit_before and not hit_after:
                regressions += 1
        rows.append({"threshold": tau, "paraphrases_fixed": helped, "false_fires": false_fires, "regressions": regressions})
    return rows


def pick_threshold(rows: List[Dict[str, Any]]) -> float:
    safe = [r for r in rows if r["false_fires"] == 0 and r["regressions"] == 0]
    if not safe:
        return max(r["threshold"] for r in rows)
    best = max(r["paraphrases_fixed"] for r in safe)
    return min(r["threshold"] for r in safe if r["paraphrases_fixed"] == best)


# %% [condenser]
_SENT = re.compile(r"(?<=[.!?])\s+")
_HAS_FACT = re.compile(r"\d|\b(?:PAY|PO|INV|ERP|TCK)-?\d+|Settings >|Invoices >|Reports >|Vendors >|Integrations")


def condense(answer: str, max_words: int = 50) -> Tuple[str, bool]:
    """Shorten an answer for a customer who prefers brief replies. Deterministic and safe: it only DROPS
    sentences (never rewrites), keeps the first sentence and every sentence with a number, ID or menu path,
    and always keeps the ticket reference. So it can't introduce a new fact."""
    body, sep, ref = answer.partition("\n\nReference: ticket")
    words = len(body.split())
    if words <= max_words + 10:
        return answer, False
    sents = [s for s in _SENT.split(body.strip()) if s]
    keep, count = [], 0
    for i, s in enumerate(sents):
        if i == 0 or _HAS_FACT.search(s):
            n = len(s.split())
            if keep and count + n > max_words:
                break
            keep.append(s)
            count += n
    short = " ".join(keep)
    return (short + (sep + ref if sep else "")), True


# %% [learner]
@dataclass
class LearnerConfig:
    """Evidence thresholds. Every adaptation needs enough consistent evidence; nothing changes on one complaint."""
    window_days: int = 30
    style_min_votes: int = 2           # consistent too_long/too_short ratings from one customer
    guidance_min_reviews: int = 3      # specialist reviews for one topic
    guidance_min_rate: float = 0.7     # share agreeing on the direction
    monitor_min_events: int = 3        # feedback events after activation before judging
    monitor_max_negative: float = 0.5  # negative share that switches an adaptation off


def _guidance_text(intent: str, direction: str, team: Optional[str]) -> str:
    desc = INTENT_DESC.get(intent, intent.replace("_", " "))
    if direction == "relax":
        return (f"For {desc}: when an excerpt fully answers the question, answer it directly and do NOT open a ticket. "
                "Escalate only if the customer says the documented wait or steps have already failed.")
    return (f"For {desc}: always escalate to {team or 'the right team'} after giving the documented facts. "
            "Never suggest the customer retries or resubmits on their own.")


class AdaptationEngine:
    def __init__(self, registry: AdaptationRegistry, comp: "p4.RagComponents", chunks: List[p4.Chunk],
                 kb_texts: Dict[str, str], cfg: Optional[LearnerConfig] = None,
                 ltm_store: Optional[p6.LongTermMemoryStore] = None, hint_threshold: float = 0.5) -> None:
        self.registry, self.comp, self.chunks, self.kb_texts = registry, comp, chunks, kb_texts
        self.cfg = cfg or LearnerConfig()
        self.ltm_store = ltm_store or p6.LongTermMemoryStore()
        self.hint_threshold = hint_threshold

    def hint_index(self) -> HintIndex:
        idx = HintIndex(self.comp.embeddings, self.chunks, self.hint_threshold)
        for a in self.registry.active("retrieval_hint"):
            idx.add(a.id, a.change["query"], a.change["doc_id"])
        return idx

    # ---- the four learning rules ----
    def learn(self, events: List[FeedbackEvent], gate_queries: List[Dict[str, Any]], now: datetime) -> List[Adaptation]:
        recent = [e for e in events if datetime.fromisoformat(e.ts) >= now - timedelta(days=self.cfg.window_days)]
        props = self._style(recent) + self._guidance(recent) + self._examples(recent) + self._hints(recent, gate_queries)
        for a in props:
            self.registry.upsert(a)
            if a.status == "active":
                self._apply_side_effects(a)
        return props

    def _style(self, events: List[FeedbackEvent]) -> List[Adaptation]:
        votes: Dict[str, Counter] = defaultdict(Counter)
        for e in events:
            if e.source == "customer" and e.user_key and e.reason in {"too_long", "too_short"}:
                votes[e.user_key][e.reason] += 1
        out = []
        for uk, c in votes.items():
            longv, shortv = c["too_long"], c["too_short"]
            if max(longv, shortv) < self.cfg.style_min_votes or longv == shortv:
                continue
            style = "brief" if longv > shortv else "detailed"
            out.append(Adaptation(
                id=style_adaptation_id(uk), kind="reply_style", scope=f"customer:{uk}", change={"reply_style": style},
                reason=f"{max(longv, shortv)} of this customer's ratings said answers were "
                       f"{'too long' if style == 'brief' else 'too short'} (against {min(longv, shortv)} the other way).",
                evidence={"too_long": longv, "too_short": shortv}, risk="low", status="active",
                gate={"reason": "low risk: per-customer, enforced by a sentence-dropping condenser that can't add facts"}))
        return out

    def _guidance(self, events: List[FeedbackEvent]) -> List[Adaptation]:
        by_intent: Dict[str, List[FeedbackEvent]] = defaultdict(list)
        for e in events:
            if e.source == "specialist" and e.kind in {"ticket_review", "answer_review"} and e.intent:
                by_intent[e.intent].append(e)
        out = []
        for intent, evs in by_intent.items():
            n = len(evs)
            relax = sum(e.verdict == "could_self_serve" or e.reason == "unnecessary_escalation" for e in evs)
            tighten = sum(e.verdict == "should_have_escalated" or e.reason == "should_escalate" for e in evs)
            confirm = sum(e.verdict == "correct_escalation" for e in evs)
            if n < self.cfg.guidance_min_reviews:
                continue
            for direction, k in (("relax", relax), ("tighten", tighten)):
                if k / n < self.cfg.guidance_min_rate:
                    continue
                team = Counter(e.expected_team for e in evs if e.expected_team).most_common(1)
                team = team[0][0] if team else None
                a = Adaptation(
                    id=f"A-guide-{intent}-{direction}", kind="intent_guidance", scope=f"intent:{intent}",
                    change={"intent": intent, "direction": direction, "team": team,
                            "guidance": _guidance_text(intent, direction, team)},
                    reason=f"{k} of {n} specialist reviews of {INTENT_DESC.get(intent, intent)} said the agent should "
                           f"{'not have escalated' if direction == 'relax' else 'have escalated'}.",
                    evidence={"reviews": n, "agree": k, "rate": round(k / n, 2), "confirmed_correct": confirm,
                              "feedback_ids": [e.feedback_id for e in evs][:8]},
                    risk="medium" if direction == "relax" else "low")
                if direction == "relax" and intent in SAFETY_LOCKED:
                    a.status, a.gate = "rejected", {"reason": f"'{intent}' is safety-locked: adaptations may make it stricter, never looser"}
                elif direction == "relax":
                    approver = next((e.approved_by for e in evs if e.approved_by), None)
                    a.status = "pending_approval"
                    a.gate = {"reason": "relaxing escalation needs a named approver", "suggested_approver": approver}
                else:
                    a.status, a.gate = "active", {"reason": "tightening is low risk: it can only add escalations"}
                out.append(a)
        return out

    def _examples(self, events: List[FeedbackEvent]) -> List[Adaptation]:
        out, per_intent = [], Counter()
        for e in events:
            if e.kind != "approved_example" or not e.approved_answer:
                continue
            per_intent[e.intent] += 1
            aid = f"A-example-{e.intent}-{per_intent[e.intent]}"
            problems = []
            if not e.approved_by:
                problems.append("no approver")
            missing = [c for c in e.approved_citations if c not in self.kb_texts]
            if missing or not e.approved_citations:
                problems.append(f"citations not in the knowledge base: {missing or 'none given'}")
            else:
                cited = " ".join(self.kb_texts[c] for c in e.approved_citations)
                bad = sorted(p3._numbers(e.approved_answer) - p3._numbers(cited))
                if bad:
                    problems.append(f"numbers not in the cited documents: {bad}")
            if per_intent[e.intent] > 3:
                problems.append("more than 3 examples for this topic")
            out.append(Adaptation(
                id=aid, kind="learned_example", scope=f"intent:{e.intent}",
                change={"intent": e.intent, "message": e.message, "answer": e.approved_answer,
                        "citations": e.approved_citations, "route": e.approved_route},
                reason=f"Specialist-approved answer for {INTENT_DESC.get(e.intent, e.intent)} after customers reported wrong answers.",
                evidence={"approved_by": e.approved_by, "feedback_id": e.feedback_id}, risk="low",
                status="rejected" if problems else "active",
                gate={"reason": "; ".join(problems) if problems else "verified against the cited documents"},
                approved_by=e.approved_by))
        return out

    def _hints(self, events: List[FeedbackEvent], gate_queries: List[Dict[str, Any]]) -> List[Adaptation]:
        out, accepted = [], [{"id": a.id, "query": a.change["query"], "doc_id": a.change["doc_id"]}
                             for a in self.registry.active("retrieval_hint")]
        training = [{"query": e.message, "relevant_docs": [e.correct_doc_id], "group": "training"}
                    for e in events if e.kind == "retrieval_correction" and e.message and e.correct_doc_id]
        queries = gate_queries + training
        for e in events:
            if e.kind != "retrieval_correction" or not e.message or not e.correct_doc_id:
                continue
            hid = f"A-hint-{hashlib.sha256(e.message.encode()).hexdigest()[:6]}"
            cand = {"id": hid, "query": e.message, "doc_id": e.correct_doc_id}
            g = hint_gate(lambda: HintIndex(self.comp.embeddings, self.chunks, self.hint_threshold), cand, accepted, queries, self.comp)
            if g["accepted"]:
                accepted.append(cand)
            out.append(Adaptation(
                id=hid, kind="retrieval_hint", scope="global", change={"query": e.message, "doc_id": e.correct_doc_id},
                reason=f"A specialist marked {e.correct_doc_id} as the right document for a query retrieval missed.",
                evidence={"feedback_id": e.feedback_id}, risk="low", status="active" if g["accepted"] else "rejected",
                gate={**g, "reason": "passed the retrieval gate" if g["accepted"] else
                      ("not needed: retrieval already finds this document" if not g["needed"] else
                       "does not fix its own query" if not g["fixes_own_query"] else "would harm other queries")}))
        return out

    # ---- side effects, approval, monitoring, rollback ----
    def _apply_side_effects(self, a: Adaptation) -> None:
        if a.kind == "reply_style":
            uk = a.scope.split(":", 1)[1]
            rec = self.ltm_store.load(uk)
            self.ltm_store.upsert_fact(rec, "reply_style", a.change["reply_style"])
            self.ltm_store.save(uk, rec)

    def approve(self, adaptation_id: str, approver: str, note: str = "approved after review") -> Adaptation:
        a = self.registry.items[adaptation_id]
        if a.status != "pending_approval":
            raise ValueError(f"{adaptation_id} is {a.status}, not pending approval")
        if a.kind == "intent_guidance" and a.change["direction"] == "relax" and a.change["intent"] in SAFETY_LOCKED:
            raise ValueError("safety-locked topics can't be relaxed, even with approval")
        return self.registry.set_status(adaptation_id, "active", approver, note)

    def rollback(self, adaptation_id: str, by: str, note: str) -> Adaptation:
        a = self.registry.set_status(adaptation_id, "rolled_back", by, note)
        if a.kind == "reply_style":
            uk = a.scope.split(":", 1)[1]
            rec = self.ltm_store.load(uk)
            if rec["facts"].get("reply_style", {}).get("value") == a.change["reply_style"]:
                del rec["facts"]["reply_style"]
                self.ltm_store.save(uk, rec)
        return a

    def monitor(self, events: List[FeedbackEvent]) -> List[Dict[str, Any]]:
        """Post-change check: feedback given while an adaptation was active. Too negative -> switch it off."""
        report = []
        for a in self.registry.active():
            evs = [e for e in events if a.id in e.adaptations_active]
            neg = sum((e.rating == "down") or (e.reason in NEGATIVE_REASONS) for e in evs)
            rate = neg / len(evs) if evs else 0.0
            decision = "keep (not enough data)" if len(evs) < self.cfg.monitor_min_events else "keep"
            if len(evs) >= self.cfg.monitor_min_events and rate >= self.cfg.monitor_max_negative:
                decision = "disabled"
                self.registry.set_status(a.id, "disabled", "monitor",
                                         f"{neg} of {len(evs)} feedback events after activation were negative")
                if a.kind == "reply_style":
                    self.rollback(a.id, "monitor", "style reverted with the adaptation")
            report.append({"adaptation": a.id, "events_after_activation": len(evs), "negative": neg,
                           "negative_rate": round(rate, 2), "decision": decision})
        return report


# %% [agent]
class AdaptiveToolBox(p6.SessionToolBox):
    """Phase 6 toolbox + feedback hints in knowledge search."""

    def __init__(self, *args: Any, hint_index: Optional[HintIndex] = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.hint_index = hint_index
        self.fired_hints: List[str] = []

    def search_knowledge_base(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        result = super().search_knowledge_base(query, top_k)
        if not self.hint_index or not self.hint_index.hints:
            return result
        qv = self.comp.embeddings.embed_query(query)
        present = {r["doc_id"] for r in result.get("results", [])}
        extra = []
        for h, sim in self.hint_index.matches(qv):
            self.fired_hints.append(h["id"])
            if h["doc_id"] in present:
                continue
            c = self.hint_index.best_chunk(h["doc_id"], qv)
            extra.append({"doc_id": c.doc_id, "version": c.version, "section": f"{c.title} > {c.section}",
                          "relevance": round(sim, 2), "text": c.text, "chunk_id": c.chunk_id,
                          "source": "feedback_hint"})
            present.add(h["doc_id"])
        if extra:
            result = {"results": extra + result.get("results", [])}
        return result


@dataclass
class AdaptiveTurnResult(p6.MemoryTurnResult):
    intent_labels: List[str] = field(default_factory=list)
    adaptations_applied: List[str] = field(default_factory=list)
    words_before_condense: int = 0
    implicit_signal: Optional[str] = None


class AdaptiveAgent(p6.MemoryAgent):
    """Phase 6 agent + active adaptations. adapt=False gives the exact Phase 6 behaviour (the 'before')."""

    def __init__(self, comp: "p4.RagComponents", prompt: p3.PromptVersion, chat_model: BaseChatModel,
                 registry: Optional[AdaptationRegistry] = None, hint_index: Optional[HintIndex] = None,
                 feedback_store: Optional[FeedbackStore] = None, adapt: bool = True, **kw: Any) -> None:
        self.registry = registry or AdaptationRegistry(path=ADAPT_DIR / "_empty.json")
        self.adapt = adapt
        self.feedback_store = feedback_store
        super().__init__(comp, prompt, chat_model, **kw)
        old = self.box
        self.box = AdaptiveToolBox(old.tools, comp, old.faults, old.session_ref,
                                   hint_index=hint_index if adapt else None)
        if kw.get("log_path", p6.MEMORY_LOG) is not None:
            self.log_path = ADAPT_LOG
        self._prev_turn: Optional[Dict[str, Any]] = None

    def _relevant(self) -> List[Adaptation]:
        if not self.adapt:
            return []
        labels = set(self._labels)
        return [a for a in self.registry.active() if a.kind in {"intent_guidance", "learned_example"}
                and a.change["intent"] in labels]

    def _extra_system(self) -> Optional[str]:
        rel = self._relevant()
        if not rel:
            return None
        lines = ["LEARNED GUIDANCE (from reviewed feedback; it never overrides the RULES above):"]
        for a in rel:
            if a.kind == "intent_guidance":
                lines.append(f"- {a.change['guidance']}")
                self._applied.append(a.id)
        examples = [a for a in rel if a.kind == "learned_example"]
        if examples:
            lines.append("SPECIALIST-APPROVED EXAMPLES (follow the reasoning; state facts only from tool results and cite them):")
            for a in examples:
                lines.append(f"Q: {a.change['message']}\nA ({a.change['route']}; cites {', '.join(a.change['citations'])}): {a.change['answer']}")
                self._applied.append(a.id)
        return "\n".join(lines)

    def _accept(self, d: Any) -> Dict[str, Any]:
        final = super()._accept(d)
        style = self.ltm.get("facts", {}).get("reply_style", {}).get("value")
        self._words_before = len(final["answer"].split())
        if self.adapt and style == "brief" and not final["guardrail_fired"]:
            final["answer"], changed = condense(final["answer"])
            if changed:
                ad = next((a.id for a in self.registry.active("reply_style") if a.scope == f"customer:{self.user_key}"),
                          "reply_style(memory)")
                self._applied.append(ad)
        return final

    def respond(self, message: Any, extra_metadata: Optional[Dict[str, Any]] = None) -> AdaptiveTurnResult:
        redacted = p3.redact_pii(message)[0] if isinstance(message, str) else ""
        self._labels = classify_intents(redacted)
        self._applied: List[str] = []
        self._words_before = 0
        signal = detect_implicit(self._prev_turn, redacted) if redacted else None
        if signal and self.feedback_store is not None and self._prev_turn:
            self.feedback_store.add({"source": "implicit", "kind": "rating", "user_key": self.user_key,
                                     "intent": self._prev_turn["labels"][0], "message": self._prev_turn["message"],
                                     "rating": signal["rating"], "reason": signal["reason"],
                                     "adaptations_active": self._prev_turn["applied"]})
        self.box.fired_hints = []
        res = super().respond(message, extra_metadata)
        applied = list(dict.fromkeys(self._applied + self.box.fired_hints))
        out = AdaptiveTurnResult(**asdict(res), intent_labels=self._labels, adaptations_applied=applied,
                                 words_before_condense=self._words_before or len(res.answer.split()),
                                 implicit_signal=f"{signal['rating']}:{signal['reason']}" if signal else None)
        self._prev_turn = {"message": redacted, "route": res.route, "labels": self._labels, "applied": applied}
        return out


# %% [explain]
def explain_turn(r: AdaptiveTurnResult, registry: AdaptationRegistry) -> str:
    if not r.adaptations_applied:
        return "No learned adaptation affected this answer; it is the standard behaviour."
    lines = ["This answer was shaped by:"]
    for aid in r.adaptations_applied:
        a = registry.items.get(aid)
        if a is None:
            lines.append(f"- {aid}: the customer's saved reply-style preference.")
        else:
            lines.append(f"- {a.id} ({a.kind}): {a.reason}")
    if r.words_before_condense > len(r.answer.split()):
        lines.append(f"  (Answer shortened from {r.words_before_condense} to {len(r.answer.split())} words by dropping sentences.)")
    return "\n".join(lines)


def change_log(registry: AdaptationRegistry, before_after: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """Human-readable record of what changed, why, the gate, the measured effect and how to undo it."""
    out = [f"# Adaptation change log (registry version {registry.version})", ""]
    for a in registry.items.values():
        out.append(f"## {a.id} [{a.status}]")
        out.append(f"- **What:** {a.kind} for {a.scope}: {json.dumps(a.change, ensure_ascii=False)[:300]}")
        out.append(f"- **Why:** {a.reason}")
        out.append(f"- **Gate / decision:** {a.gate.get('reason', '')}" + (f" (approved by {a.approved_by})" if a.approved_by else ""))
        if before_after and a.id in before_after:
            ba = before_after[a.id]
            out.append(f"- **Measured effect:** pass {ba['before']} -> {ba['after']} on {ba['cases']} held-out case(s)")
        out.append(f"- **Undo:** `engine.rollback('{a.id}', by=..., note=...)`")
        out.append(f"- **History:** " + " | ".join(a.history))
        out.append("")
    return "\n".join(out)


# %% [experiments]
def load_feedback_yaml(path: Path) -> List[Dict[str, Any]]:
    data = p3.load_yaml(path) or []
    if not isinstance(data, list):
        raise DataLoadError(f"{path.name} must be a list of feedback records.")
    return data


def load_adaptation_cases(path: Path = EVAL_DIR / "adaptation_cases.yaml") -> List[Dict[str, Any]]:
    cases = p3.load_yaml(path) or []
    for c in cases:
        c["target"] = resolve_adaptation_ref(str(c.get("target", "")))
        for k, v in {"must_include": [], "must_exclude": [], "prior_turns": [], "safety_critical": False,
                     "max_words": None, "split": "phase7", "category": "adaptation"}.items():
            c.setdefault(k, v)
    return cases


def run_before_after(cases: List[Dict[str, Any]], comp: "p4.RagComponents", prompt: p3.PromptVersion,
                     registry: AdaptationRegistry, hint_index: HintIndex, ltm_after: p6.LongTermMemoryStore,
                     ltm_before: p6.LongTermMemoryStore, results_path: Path, resume: bool = True) -> List[Dict[str, Any]]:
    done = {(r["variant"], r["case_id"]) for r in p3.read_jsonl(results_path)} if resume else set()
    todo = [(v, c) for v in ("before", "after") for c in cases if (v, c["id"]) not in done]
    print(f"{len(todo)} runs to do.")
    for i, (variant, case) in enumerate(todo, 1):
        planner = None if comp.settings.dry_run else p3.make_chat_model(comp.settings)
        agent = AdaptiveAgent(comp, prompt, p5.make_tool_model(comp.settings), registry=registry,
                              hint_index=hint_index, adapt=(variant == "after"), planner_model=planner,
                              store=ltm_after if variant == "after" else ltm_before, user_id=case["user_id"])
        for prior in case["prior_turns"]:
            agent.respond(prior)
        r = agent.respond(case["message"], extra_metadata={"case_id": case["id"], "experiment": "phase7", "variant": variant})
        sc = p3.score(case, r.answer, r.route, r.error, r.guardrail_fired)
        body = r.answer.split("\n\nReference: ticket")[0]
        words = len(body.split())
        style_ok = case["max_words"] is None or words <= case["max_words"]
        row = {"variant": variant, "case_id": case["id"], "target": case["target"], "user_id": case["user_id"],
               "safety_critical": case["safety_critical"], "expected_routes": case["expected_routes"],
               "route": r.route, "passed": sc["passed"] and style_ok, "answer_ok": sc["passed"], "style_ok": style_ok,
               "failure_type": sc["failure_type"] or ("too_long" if not style_ok else None), "words": words,
               "tickets": r.ticket_ids, "tools": r.tools_attempted, "adaptations_applied": r.adaptations_applied,
               "intent_labels": r.intent_labels, "answer": p3.redact_pii(r.answer)[0], "error": r.error,
               "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "dry_run": comp.settings.dry_run}
        p3.append_jsonl(results_path, row)
        print(f"[{i}/{len(todo)}] {variant:<6} {case['id']:<26} {r.route:<10} {'PASS' if row['passed'] else 'FAIL'} "
              f"words={words:<4} adapted={r.adaptations_applied}")
        if r.error in p3.FATAL_ERRORS:
            raise SetupError("Stopping: fatal LLM error. Check OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL.")
    return p3.read_jsonl(results_path)
