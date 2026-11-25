"""
config.py: one typed, validated configuration for the deployed service, plus startup checks ("doctor").

Every setting comes from environment variables (.env in development, real env vars in the cloud).
A bad value fails at startup with a clear message, never halfway through a customer conversation.
"""
import importlib
import importlib.metadata as md
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


class ConfigError(ValueError):
    """A setting is missing or invalid. The message says which and how to fix it."""


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}.")


def _num(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc
    if not lo <= v <= hi:
        raise ConfigError(f"{name}={v} is outside the allowed range {lo}-{hi}.")
    return v


@dataclass
class AppConfig:
    app_name: str = "PayLedger Support Assistant"
    app_version: str = "0.0.0"
    environment: str = "dev"                 # dev | staging | prod
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = "INFO"
    log_max_mb: float = 5.0
    log_backups: int = 3
    request_timeout_s: float = 45.0          # hard deadline for one customer message
    llm_retries: int = 2                     # extra attempts for transient LLM errors (on top of the SDK's own)
    retry_base_s: float = 0.5
    breaker_failures: int = 3                # failures within the window that open a circuit
    breaker_window_s: float = 60.0
    breaker_cooldown_s: float = 30.0
    session_idle_s: float = 1800.0           # in-memory sessions dropped after this idle time
    max_sessions: int = 500
    planning: bool = True
    memory: bool = True
    adaptation: bool = True
    faults: str = ""                         # fault injection, e.g. "llm:timeout@2,vector_store:down" (never in prod)
    memory_dir: Path = BASE_DIR / "memory_store"
    feedback_path: Path = BASE_DIR / "feedback" / "feedback.jsonl"
    registry_path: Path = BASE_DIR / "adaptations" / "registry.json"
    dry_run: bool = False                    # mirrored from the LLM settings for display

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def _path(name: str, default: Path) -> Path:
    """Relative paths are resolved against the project folder, not the launch directory."""
    raw = os.getenv(name, "").strip()
    p = Path(raw) if raw else default
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def load_config(env_file: Optional[Path] = None) -> AppConfig:
    env_file = env_file or BASE_DIR / ".env"
    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file, override=False)
    version_file = BASE_DIR / "VERSION"
    cfg = AppConfig(
        app_version=version_file.read_text().strip() if version_file.exists() else "0.0.0",
        environment=os.getenv("APP_ENV", "dev").strip().lower(),
        log_dir=_path("LOG_DIR", BASE_DIR / "logs"),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        log_max_mb=_num("LOG_MAX_MB", 5.0, 0.1, 500),
        log_backups=int(_num("LOG_BACKUPS", 3, 0, 50)),
        request_timeout_s=_num("REQUEST_TIMEOUT_S", 45.0, 1, 600),
        llm_retries=int(_num("LLM_APP_RETRIES", 2, 0, 5)),
        retry_base_s=_num("RETRY_BASE_S", 0.5, 0, 10),
        breaker_failures=int(_num("BREAKER_FAILURES", 3, 1, 100)),
        breaker_window_s=_num("BREAKER_WINDOW_S", 60, 1, 3600),
        breaker_cooldown_s=_num("BREAKER_COOLDOWN_S", 30, 0, 3600),
        session_idle_s=_num("SESSION_IDLE_S", 1800, 60, 86400),
        max_sessions=int(_num("MAX_SESSIONS", 500, 1, 100000)),
        planning=_bool("FEATURE_PLANNING", True),
        memory=_bool("FEATURE_MEMORY", True),
        adaptation=_bool("FEATURE_ADAPTATION", True),
        faults=os.getenv("FAULTS", "").strip(),
        memory_dir=_path("MEMORY_DIR", BASE_DIR / "memory_store"),
        feedback_path=_path("FEEDBACK_PATH", BASE_DIR / "feedback" / "feedback.jsonl"),
        registry_path=_path("REGISTRY_PATH", BASE_DIR / "adaptations" / "registry.json"),
    )
    if cfg.environment not in {"dev", "staging", "prod"}:
        raise ConfigError(f"APP_ENV must be dev, staging or prod, got {cfg.environment!r}.")
    if cfg.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError(f"LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR, got {cfg.log_level!r}.")
    if cfg.faults and cfg.environment == "prod":
        raise ConfigError("FAULTS (fault injection) is not allowed when APP_ENV=prod.")
    return cfg


# ---------------------------------------------------------------- doctor
@dataclass
class Check:
    name: str
    status: str   # ok | warn | fail
    detail: str


REQUIRED_PACKAGES = ["langchain-core", "langchain-openai", "langgraph", "openai", "pinecone", "pydantic", "pyyaml",
                     "python-dotenv", "numpy", "pandas", "streamlit"]
REQUIRED_FILES = ["data/kb", "data/mock_db.json", "data/kb_policies.json", "prompts/phase6/v7_memory_planning.yaml",
                  "prompts/phase5/v6_tools.yaml", "eval/rag_cases.yaml"]


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return True
    except OSError:
        return False


def doctor(cfg: AppConfig, online: bool = False) -> List[Check]:
    """Startup checks. 'fail' means the service must not start; 'warn' means it starts in a degraded mode."""
    out: List[Check] = []
    v = sys.version_info
    out.append(Check("python", "ok" if (v.major, v.minor) == (3, 10) else "warn",
                     f"{v.major}.{v.minor}.{v.micro} (built and tested on 3.10)"))
    for pkg in REQUIRED_PACKAGES:
        try:
            out.append(Check(f"package {pkg}", "ok", md.version(pkg)))
        except md.PackageNotFoundError:
            out.append(Check(f"package {pkg}", "fail", "not installed: pip install -r requirements.txt"))
    for f in REQUIRED_FILES:
        out.append(Check(f"file {f}", "ok" if (BASE_DIR / f).exists() else "fail",
                         "present" if (BASE_DIR / f).exists() else "missing: keep the project folder intact"))
    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    has_pc = bool(os.getenv("PINECONE_API_KEY", "").strip())
    out.append(Check("OPENAI_API_KEY", "ok" if has_openai else "warn",
                     "set" if has_openai else "not set: DRY RUN mode (scripted answers, keyword search)"))
    out.append(Check("PINECONE_API_KEY", "ok" if has_pc else "warn",
                     "set" if has_pc else "not set: vectors kept in memory (rebuilt at every start)"))
    out.append(Check("LANGSMITH_API_KEY", "ok" if os.getenv("LANGSMITH_API_KEY", "").strip() else "warn",
                     "tracing on" if os.getenv("LANGSMITH_API_KEY", "").strip() else "not set: no LangSmith tracing (local logs still on)"))
    for label, path in [("log dir", cfg.log_dir), ("memory dir", cfg.memory_dir), ("feedback dir", cfg.feedback_path.parent),
                        ("adaptations dir", cfg.registry_path.parent)]:
        ok = _writable(path)
        out.append(Check(f"writable {label}", "ok" if ok else ("fail" if label == "log dir" else "warn"),
                         str(path) if ok else f"{path} not writable: feature runs without persistence"))
    try:
        import rag_agent as p4
        docs = p4.load_corpus()
        out.append(Check("knowledge base", "ok", f"{len(docs)} documents, fingerprint {p4.corpus_fingerprint(docs)}"))
    except Exception as exc:  # noqa: BLE001 - report any loader problem
        out.append(Check("knowledge base", "fail", f"{type(exc).__name__}: {str(exc)[:160]}"))
    if cfg.faults:
        out.append(Check("fault injection", "warn", f"FAULTS={cfg.faults!r} (testing only)"))
    if online:
        out += _online_checks(has_openai, has_pc)
    return out


def _online_checks(has_openai: bool, has_pc: bool) -> List[Check]:
    out = []
    if has_openai:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None, timeout=10, max_retries=0)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
            out.append(Check("OpenAI reachable", "ok", f"model {model} answered"))
        except Exception as exc:  # noqa: BLE001
            out.append(Check("OpenAI reachable", "fail", f"{type(exc).__name__}: check key, base URL and model"))
    if has_pc:
        try:
            from pinecone import Pinecone
            Pinecone(api_key=os.getenv("PINECONE_API_KEY")).list_indexes()
            out.append(Check("Pinecone reachable", "ok", "listed indexes"))
        except Exception as exc:  # noqa: BLE001
            out.append(Check("Pinecone reachable", "fail", f"{type(exc).__name__}: check key and network"))
    return out
