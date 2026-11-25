"""
baseline_agent.py
Phase 2: Baseline AI Support Resolution Agent (rules + templates).

Scenario : Fintech B2B SaaS (payments, payouts, invoicing, ERP sync).
Approach : Regex rules pick ONE intent, a template fills in KB text and
           mock read-only tool data. No LLM, no vector DB, no memory.
           It is kept deliberately simple so its limits are visible.

Run it:
    python baseline_agent.py            # interactive chat in the terminal
    python baseline_agent.py --demo     # run the sample interactions
    python baseline_agent.py --tenant T-001

Target runtime: Python 3.10 (Vocareum).
"""

# %% [imports]
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # python-dotenv is optional for the baseline (no API keys needed yet)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# In a notebook __file__ does not exist, so fall back to the working directory.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
KB_PATH = DATA_DIR / "kb_policies.json"
DB_PATH = DATA_DIR / "mock_db.json"
INTERACTION_LOG = LOG_DIR / "interactions.jsonl"
APP_LOG = LOG_DIR / "baseline_agent.log"

MAX_INPUT_CHARS = 2000
REFERENCE_DATE = date(2026, 11, 23)  # fixed "today" so SLA checks are reproducible
SLA_BUSINESS_DAYS = {"ACH": 3, "SAME_DAY_ACH": 1, "SWIFT": 5}
ROUTES = ("resolved", "needs_info", "escalated", "refused", "unresolved")


# %% [config]
class DataLoadError(RuntimeError):
    """Raised when a data file is missing or malformed."""


class InputError(ValueError):
    """Raised when the user message cannot be processed."""


def load_environment() -> Dict[str, bool]:
    """Load .env (if present) and report WHICH keys exist, never their values.

    The baseline needs no keys. Later phases need OpenAI, Pinecone and
    LangSmith keys, so we check them here to catch setup problems early.
    """
    expected = ["OPENAI_API_KEY", "PINECONE_API_KEY", "LANGSMITH_API_KEY"]
    env_file = BASE_DIR / ".env"
    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file, override=False)
    return {key: bool(os.getenv(key)) for key in expected}


def setup_logging() -> logging.Logger:
    """App log for metadata only (route, intent, latency). Never message text."""
    logger = logging.getLogger("baseline_agent")
    if logger.handlers:  # avoid duplicate handlers when a notebook cell re-runs
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(APP_LOG, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    except OSError:
        handler = logging.NullHandler()  # read-only filesystem: keep running, skip logs
    logger.addHandler(handler)
    return logger


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"Missing data file: {path}. Keep the data/ folder next to the notebook or script."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid JSON in {path.name} at line {exc.lineno}.") from exc


def load_knowledge_base(path: Path = KB_PATH) -> Dict[str, Dict[str, Any]]:
    """Load approved policy docs. Every answer must cite one of these."""
    docs = load_json(path)
    required = {"doc_id", "title", "version", "effective_date", "text"}
    kb: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        missing = required - set(doc)
        if missing:
            raise DataLoadError(f"KB doc {doc.get('doc_id', '?')} is missing fields: {sorted(missing)}")
        kb[doc["doc_id"]] = doc
    return kb


def load_mock_db(path: Path = DB_PATH) -> Dict[str, Any]:
    db = load_json(path)
    for table in ("payments", "payouts", "invoices", "error_codes"):
        if table not in db:
            raise DataLoadError(f"mock_db.json is missing the '{table}' table.")
    return db


# %% [pii]
# Order matters: specific patterns run before generic digit runs.
PII_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PAN_IN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")),
    ("ACCOUNT_NO", re.compile(r"\b\d{9,18}\b")),
]


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Replace PII with [LABEL] tags. Returns redacted text and counts per label."""
    counts: Dict[str, int] = {}
    for label, pattern in PII_PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


# %% [entities]
ID_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "payment_id": re.compile(r"\bPAY-\d{4,6}\b", re.IGNORECASE),
    "payout_id": re.compile(r"\bPO-\d{3,6}\b", re.IGNORECASE),
    "invoice_id": re.compile(r"\bINV-\d{3,6}\b", re.IGNORECASE),
    "error_code": re.compile(r"\bERP-\d{3}\b", re.IGNORECASE),
}


def extract_entities(text: str) -> Dict[str, str]:
    """Pull the first ID of each type from the message (strict formats only)."""
    found: Dict[str, str] = {}
    for name, pattern in ID_PATTERNS.items():
        match = pattern.search(text)
        if match:
            found[name] = match.group(0).upper()
    return found


# %% [tools]
class MockTools:
    """Read-only lookups that stand in for the platform's internal APIs.

    Tenant isolation lives HERE, not in the prompt: a record from another
    tenant is treated exactly like a missing record, so nothing leaks.
    """

    def __init__(self, db: Dict[str, Any], tenant_id: str) -> None:
        self.db = db
        self.tenant_id = tenant_id
        self.tickets: List[Dict[str, str]] = []

    def _lookup(self, table: str, key: str) -> Optional[Dict[str, Any]]:
        record = self.db.get(table, {}).get(key)
        if not record or record.get("tenant_id") != self.tenant_id:
            return None
        return record

    def get_payment_status(self, payment_id: str) -> Optional[Dict[str, Any]]:
        return self._lookup("payments", payment_id)

    def get_payout_status(self, payout_id: str) -> Optional[Dict[str, Any]]:
        return self._lookup("payouts", payout_id)

    def get_invoice(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        return self._lookup("invoices", invoice_id)

    def lookup_error_code(self, code: str) -> Optional[str]:
        return self.db.get("error_codes", {}).get(code)

    def create_ticket(self, category: str, priority: str, summary: str) -> str:
        """Simulates handing the case to a human queue. Summary is pre-redacted."""
        ticket_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
        self.tickets.append(
            {"ticket_id": ticket_id, "category": category, "priority": priority, "summary": summary}
        )
        return ticket_id


def business_days_between(start: date, end: date) -> int:
    """Business days after `start` up to and including `end` (Mon-Fri only)."""
    days, current = 0, start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days


# %% [rules]
def _compile(patterns: List[str]) -> List["re.Pattern[str]"]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# 1) Unsafe or policy-violating requests -> refuse.
UNSAFE_RULES: List[Tuple[str, List["re.Pattern[str]"]]] = [
    ("prompt_injection", _compile([
        r"ignore (all |your |the |any )?(previous |prior )?(rules|instructions|policies|policy)",
        r"\byou are now\b", r"\bdeveloper mode\b", r"\bpretend (to be|you are)\b",
    ])),
    ("bank_detail_change", _compile([
        r"\b(change|update|switch|replace|modify|edit)\b.{0,40}\bbank\b",
        r"\bnew bank (account|details)\b",
    ])),
    ("cross_tenant_data", _compile([
        r"\b(other|another|all) (customers?|clients?|tenants?)\b",
    ])),
    ("credential_request", _compile([
        r"\b(full|complete) (card|account) number\b", r"\b(cvv|otp|password)\b",
    ])),
]

# 2) Sensitive cases -> escalate to a human straight away.
SENSITIVE_RULES: List[Tuple[str, List["re.Pattern[str]"]]] = [
    ("fraud_or_takeover", _compile([
        r"\bfraud", r"\bhack(ed)?\b", r"\bunauthori[sz]ed\b", r"\bstolen\b", r"\bscam",
    ])),
    ("legal_or_regulatory", _compile([
        r"\blawyer\b", r"\blegal action\b", r"\bsue\b", r"\bregulator\b", r"\bcomplaint to\b",
    ])),
    ("dispute", _compile([r"\bchargeback\b", r"\bdispute\b"])),
]

# 3) Intents. FIRST match wins, so the order is part of the logic.
INTENT_RULES: List[Tuple[str, List["re.Pattern[str]"]]] = [
    ("duplicate_charge", _compile([
        r"charged twice", r"double[- ]charged", r"duplicate (charge|payment)", r"\brefund",
    ])),
    ("erp_error", _compile([
        r"\bERP-\d{3}\b", r"\b(netsuite|quickbooks|erp)\b.{0,30}\b(sync|error|fail)", r"\bsync (failed|error)",
    ])),
    ("payment_status", _compile([
        r"\bPAY-\d+",
        r"\bpayment\b.{0,40}\b(status|processing|pending|stuck|land|arrive|failed|where)",
        r"\b(status|where)\b.{0,30}\bpayment\b",
    ])),
    ("fees", _compile([r"\bfees?\b", r"\bpricing\b", r"\bhow much\b.{0,30}\bcost", r"\bcharge for\b"])),
    ("payout_status", _compile([r"\bPO-\d+", r"\bpayouts?\b"])),
    ("settlement_time", _compile([r"\bhow long\b", r"\bsettle", r"\bcut-?off\b"])),
    ("human_request", _compile([r"\b(human|real person|representative|talk to an agent)\b"])),
    ("greeting", _compile([r"^\s*(hi|hello|hey)\b"])),
]

INTERNATIONAL_WORDS = _compile([r"\b(eur|gbp|international|fx|swift|cross-border|foreign)\b"])
WAIVER_WORDS = _compile([r"\b(waive|waiver|discount|negotiate|lower)\b"])


def match_rules(text: str, rules: List[Tuple[str, List["re.Pattern[str]"]]]) -> List[str]:
    """Return the names of ALL rules with at least one matching pattern."""
    return [name for name, patterns in rules if any(p.search(text) for p in patterns)]


def first_intent(text: str) -> Optional[str]:
    matches = match_rules(text, INTENT_RULES)
    return matches[0] if matches else None


# %% [agent]
@dataclass
class AgentResponse:
    text: str
    route: str  # one of ROUTES
    intent: str
    matched_rules: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    ticket_id: Optional[str] = None


FALLBACK_TEXT = (
    "Sorry, I didn't understand that. I can help with:\n"
    "- payment status (share the ID, e.g. PAY-10482)\n"
    "- payout status (e.g. PO-7781)\n"
    "- fees and settlement times\n"
    "- ERP sync errors (e.g. ERP-409)\n"
    "- duplicate charges (e.g. INV-2231)\n"
    "Type 'agent' to reach a human."
)


class BaselineSupportAgent:
    """Rules + templates. Stateless: each message is handled on its own."""

    def __init__(self, kb: Dict[str, Dict[str, Any]], tools: MockTools,
                 logger: Optional[logging.Logger] = None) -> None:
        self.kb = kb
        self.tools = tools
        self.logger = logger or setup_logging()
        self.session_id = uuid.uuid4().hex[:12]  # random, not linked to the user
        self.turn = 0

    # ---------- helpers ----------
    def cite(self, doc_id: str) -> str:
        doc = self.kb[doc_id]
        return f"[{doc_id} v{doc['version']}, effective {doc['effective_date']}]"

    def policy(self, doc_id: str) -> str:
        return self.kb[doc_id]["text"]

    @staticmethod
    def _validate(message: Any) -> str:
        if message is None or not isinstance(message, str):
            raise InputError("Please type your question as text.")
        cleaned = message.strip()
        if not cleaned:
            raise InputError("Your message was empty. What can I help you with?")
        if len(cleaned) > MAX_INPUT_CHARS:
            raise InputError(f"Your message is too long. Please keep it under {MAX_INPUT_CHARS} characters.")
        return cleaned

    # ---------- main entry point ----------
    def respond(self, message: Any) -> AgentResponse:
        start = time.perf_counter()
        self.turn += 1
        try:
            text = self._validate(message)
            response = self._route(text)
        except InputError as exc:
            response = AgentResponse(text=str(exc), route="needs_info", intent="invalid_input")
        except Exception as exc:  # never crash in front of a user
            # Log the error TYPE only: exception messages can contain user text.
            self.logger.error("unhandled_error session=%s type=%s", self.session_id, type(exc).__name__)
            response = AgentResponse(
                text="Sorry, something went wrong on my side. I've passed this to a human agent.",
                route="escalated", intent="system_error",
            )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        self._log_interaction(message, response, latency_ms)
        return response

    def _route(self, text: str) -> AgentResponse:
        # 1) Safety first: refuse unsafe requests before anything else.
        unsafe = match_rules(text, UNSAFE_RULES)
        if unsafe:
            return self._refuse(text, unsafe)
        # 2) Sensitive cases go straight to a human.
        sensitive = match_rules(text, SENSITIVE_RULES)
        if sensitive:
            return self._escalate_sensitive(text, sensitive)
        # 3) One intent, one template.
        intent = first_intent(text)
        entities = extract_entities(text)
        handlers = {
            "duplicate_charge": self._handle_duplicate_charge,
            "erp_error": self._handle_erp_error,
            "payment_status": self._handle_payment_status,
            "fees": self._handle_fees,
            "payout_status": self._handle_payout_status,
            "settlement_time": self._handle_settlement,
            "human_request": self._handle_human_request,
            "greeting": self._handle_greeting,
        }
        if intent is None:
            return AgentResponse(text=FALLBACK_TEXT, route="unresolved", intent="fallback")
        response = handlers[intent](text, entities)
        response.matched_rules.insert(0, f"intent:{intent}")
        return response

    # ---------- safety handlers ----------
    def _refuse(self, text: str, rules: List[str]) -> AgentResponse:
        lines = ["I can't help with that request."]
        citations: List[str] = []
        ticket = None
        if "bank_detail_change" in rules:
            lines.append("For your security, bank details can only be changed by an account admin in "
                         "Settings > Payout Accounts with two-factor verification. "
                         f"Support never changes them over chat. {self.cite('POL-SEC-004')}")
            citations.append("POL-SEC-004")
            ticket = self.tools.create_ticket("security", "high", "Customer asked to change payout bank details in chat.")
        if "cross_tenant_data" in rules:
            lines.append("I can only share information about your own account. "
                         f"{self.cite('POL-PRIV-005')}")
            citations.append("POL-PRIV-005")
        if "credential_request" in rules:
            lines.append("Never share passwords, OTPs, CVVs or full card numbers, and we will never ask for them. "
                         f"{self.cite('POL-REFUND-003')}")
            citations.append("POL-REFUND-003")
        if "prompt_injection" in rules:
            lines.append("I follow fixed support policies and can't change or ignore them.")
        if ticket:
            lines.append(f"I've alerted our security team (ticket {ticket}).")
        return AgentResponse("\n".join(lines), "refused", "unsafe_request",
                             [f"unsafe:{r}" for r in rules], citations, ticket)

    def _escalate_sensitive(self, text: str, rules: List[str]) -> AgentResponse:
        ticket = self.tools.create_ticket("risk_disputes", "urgent", f"Sensitive case: {', '.join(rules)}.")
        reply = ("This needs our Risk and Disputes team, so I've escalated it as urgent "
                 f"(ticket {ticket}). They respond within 1 business day. {self.cite('POL-DISPUTE-006')}")
        return AgentResponse(reply, "escalated", "sensitive_case",
                             [f"sensitive:{r}" for r in rules], ["POL-DISPUTE-006"], ticket)

    # ---------- intent handlers ----------
    def _handle_payment_status(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        pid = ent.get("payment_id")
        if not pid:
            return AgentResponse("Please share the payment ID (format PAY-12345) so I can check it.",
                                 "needs_info", "payment_status")
        record = self.tools.get_payment_status(pid)
        if record is None:
            return AgentResponse(f"I couldn't find payment {pid} in your account. Please check the ID.",
                                 "needs_info", "payment_status")
        status = record["status"]
        if status == "completed":
            return AgentResponse(f"Payment {pid} completed on {record['completed']}.",
                                 "resolved", "payment_status")
        if status == "failed":
            code = record.get("return_code") or "unknown"
            reply = (f"Payment {pid} failed with bank return code {code}. {self.policy('KB-PAYFAIL-007')} "
                     f"{self.cite('KB-PAYFAIL-007')}")
            return AgentResponse(reply, "resolved", "payment_status", citations=["KB-PAYFAIL-007"])
        # processing: compare age with the SLA for this payment method
        created = date.fromisoformat(record["created"])
        sla = SLA_BUSINESS_DAYS.get(record["method"], 3)
        age = business_days_between(created, REFERENCE_DATE)
        if age > sla:
            ticket = self.tools.create_ticket("payments_ops", "high", f"{pid} processing {age} business days, SLA {sla}.")
            reply = (f"Payment {pid} ({record['method']}) has been processing for {age} business days, "
                     f"which is past the {sla}-day SLA. I've escalated it to Payments Operations "
                     f"(ticket {ticket}). {self.cite('POL-SETTLE-002')}")
            return AgentResponse(reply, "escalated", "payment_status", citations=["POL-SETTLE-002"], ticket_id=ticket)
        reply = (f"Payment {pid} ({record['method']}) is processing, day {age} of a {sla}-business-day SLA. "
                 f"No action needed yet. {self.cite('POL-SETTLE-002')}")
        return AgentResponse(reply, "resolved", "payment_status", citations=["POL-SETTLE-002"])

    def _handle_payout_status(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        po = ent.get("payout_id")
        if not po:
            return AgentResponse("Please share the payout ID (format PO-1234) so I can check it.",
                                 "needs_info", "payout_status")
        record = self.tools.get_payout_status(po)
        if record is None:
            return AgentResponse(f"I couldn't find payout {po} in your account. Please check the ID.",
                                 "needs_info", "payout_status")
        if record["status"] == "failed":
            reply = (f"Payout {po} failed with return code {record.get('return_code')}. "
                     f"{self.policy('KB-PAYFAIL-007')} {self.cite('KB-PAYFAIL-007')}")
            return AgentResponse(reply, "resolved", "payout_status", citations=["KB-PAYFAIL-007"])
        return AgentResponse(f"Payout {po} is {record['status']} (completed {record.get('completed')}).",
                             "resolved", "payout_status")

    def _handle_fees(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        cite = self.cite("POL-FEES-001")
        if any(p.search(text) for p in INTERNATIONAL_WORDS):
            reply = ("International payouts (EUR, GBP and other non-USD) cost a flat USD 5.00 per payout "
                     f"plus a 1.0% FX markup. {cite}")
        else:
            reply = ("Fees: domestic ACH USD 0.50 per payout, same-day ACH USD 2.00, international USD 5.00 "
                     f"plus 1.0% FX markup, card payments 2.9%. {cite}")
        if any(p.search(text) for p in WAIVER_WORDS):
            ticket = self.tools.create_ticket("account_management", "normal", "Customer asked about a fee waiver.")
            reply += ("\nFee waivers aren't available through support. I've asked your account manager "
                      f"to contact you (ticket {ticket}).")
            return AgentResponse(reply, "escalated", "fees", citations=["POL-FEES-001"], ticket_id=ticket)
        return AgentResponse(reply, "resolved", "fees", citations=["POL-FEES-001"])

    def _handle_erp_error(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        code = ent.get("error_code")
        if not code:
            return AgentResponse("Please share the error code from Integrations > Sync Log (e.g. ERP-409).",
                                 "needs_info", "erp_error")
        doc_id = self.tools.lookup_error_code(code)
        if doc_id is None or doc_id not in self.kb:
            ticket = self.tools.create_ticket("integrations", "normal", f"Undocumented ERP error {code}.")
            return AgentResponse(f"I don't have a documented fix for {code}. I've opened ticket {ticket} "
                                 "with our integrations team.", "escalated", "erp_error", ticket_id=ticket)
        return AgentResponse(f"{self.policy(doc_id)} {self.cite(doc_id)}", "resolved", "erp_error",
                             citations=[doc_id])

    def _handle_duplicate_charge(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        inv = ent.get("invoice_id")
        if not inv:
            return AgentResponse("Please share the invoice ID (format INV-1234) so I can check the charges.",
                                 "needs_info", "duplicate_charge")
        record = self.tools.get_invoice(inv)
        if record is None:
            return AgentResponse(f"I couldn't find invoice {inv} in your account. Please check the ID.",
                                 "needs_info", "duplicate_charge")
        charges = record["charges"]
        if len(charges) > 1 and len({c["amount"] for c in charges}) == 1:
            ticket = self.tools.create_ticket("billing", "high", f"Possible duplicate charge on {inv}.")
            reply = (f"I can see {len(charges)} charges of {record['currency']} {charges[0]['amount']:,.2f} on {inv}. "
                     "I can't issue refunds myself, so I've sent this to Billing as high priority "
                     f"(ticket {ticket}). Once verified, refunds take 5-7 business days. {self.cite('POL-REFUND-003')}")
            return AgentResponse(reply, "escalated", "duplicate_charge", citations=["POL-REFUND-003"], ticket_id=ticket)
        return AgentResponse(f"I see a single charge on {inv}, so there's no duplicate. "
                             f"{self.cite('POL-REFUND-003')}", "resolved", "duplicate_charge",
                             citations=["POL-REFUND-003"])

    def _handle_settlement(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        return AgentResponse(f"{self.policy('POL-SETTLE-002')} {self.cite('POL-SETTLE-002')}",
                             "resolved", "settlement_time", citations=["POL-SETTLE-002"])

    def _handle_human_request(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        ticket = self.tools.create_ticket("general", "normal", "Customer asked for a human agent.")
        return AgentResponse(f"I've passed you to a human agent (ticket {ticket}).", "escalated",
                             "human_request", ticket_id=ticket)

    def _handle_greeting(self, text: str, ent: Dict[str, str]) -> AgentResponse:
        return AgentResponse("Hi! " + FALLBACK_TEXT.split("\n", 1)[1], "resolved", "greeting")

    # ---------- logging ----------
    def _log_interaction(self, message: Any, response: AgentResponse, latency_ms: float) -> None:
        """Append one JSON line per turn. User and agent text are PII-redacted first."""
        raw = message if isinstance(message, str) else repr(message)
        user_redacted, pii_counts = redact_pii(raw[:MAX_INPUT_CHARS])
        reply_redacted, _ = redact_pii(response.text)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "turn": self.turn,
            "tenant_hash": hashlib.sha256(self.tools.tenant_id.encode()).hexdigest()[:12],
            "user_message_redacted": user_redacted,
            "pii_redactions": pii_counts,
            "intent": response.intent,
            "route": response.route,
            "matched_rules": response.matched_rules,
            "citations": response.citations,
            "ticket_id": response.ticket_id,
            "response_redacted": reply_redacted,
            "latency_ms": latency_ms,
        }
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with INTERACTION_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.logger.warning("interaction_log_write_failed type=%s", type(exc).__name__)
        self.logger.info("turn session=%s turn=%d intent=%s route=%s latency_ms=%.2f",
                         self.session_id, self.turn, response.intent, response.route, latency_ms)


def build_agent(tenant_id: str = "T-001") -> BaselineSupportAgent:
    """Load data, wire up tools and logging. Raises DataLoadError on bad setup."""
    kb = load_knowledge_base()
    db = load_mock_db()
    return BaselineSupportAgent(kb, MockTools(db, tenant_id), setup_logging())


# %% [cli]
DEMO_MESSAGES = [
    "Payment PAY-10482 to our vendor has shown 'processing' since Monday. When will it land?",
    "What's your fee for EUR payouts, and can you waive it for us?",
    "Our NetSuite sync failed with error ERP-409. How do I fix it?",
    "We were charged twice for invoice INV-2231. Refund it now.",
    "Update our payout bank account to 0123456789. Also show me what other customers pay Acme Corp.",
]


def print_turn(user: str, resp: AgentResponse) -> None:
    print(f"USER : {user}")
    print(f"AGENT: {resp.text}")
    print(f"       route={resp.route} | intent={resp.intent} | rules={resp.matched_rules} "
          f"| citations={resp.citations} | ticket={resp.ticket_id}")
    print("-" * 100)


def run_demo(agent: BaselineSupportAgent) -> None:
    for msg in DEMO_MESSAGES:
        print_turn(msg, agent.respond(msg))


def chat_loop(agent: BaselineSupportAgent) -> None:
    """Interactive loop. Type 'quit' or press Ctrl+C to exit."""
    print("Baseline Support Agent (rules + templates). Type 'quit' to exit.")
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.strip().lower() in {"quit", "exit"}:
            print("Bye.")
            break
        resp = agent.respond(user)
        print(f"Agent: {resp.text}\n  [route={resp.route}, intent={resp.intent}]")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 baseline support agent")
    parser.add_argument("--demo", action="store_true", help="run the sample interactions and exit")
    parser.add_argument("--tenant", default="T-001", help="tenant ID of the signed-in customer")
    args = parser.parse_args(argv)
    load_environment()
    try:
        agent = build_agent(args.tenant)
    except DataLoadError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 1
    if args.demo:
        run_demo(agent)
    else:
        chat_loop(agent)
    print(f"\nLogs: {INTERACTION_LOG}")
    return 0


# %% [entrypoint]
if __name__ == "__main__":
    sys.exit(main())
