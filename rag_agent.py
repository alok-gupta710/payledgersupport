"""
rag_agent.py
Phase 4: Knowledge & Retrieval for the Support Resolution Agent.

documents (markdown + metadata) -> chunks -> embeddings -> vector store (Pinecone or in-memory)
-> LangChain retriever (metadata filter: active versions only) -> relevance threshold
-> excerpts with citations in the LLM turn -> verifier checks every number against cited excerpts.

Reuses Phase 3 (llm_agent.py) for redaction, ID parsing, read-only tools, the output schema,
the structured LLM wrapper and scoring. Target: Python 3.10 (Vocareum). Keys in .env.
"""

# %% [imports]
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(BASE_DIR) not in sys.path:  # make the Phase 3 module importable in any Jupyter setup
    sys.path.insert(0, str(BASE_DIR))

import llm_agent as p3  # Phase 3 building blocks
from llm_agent import DataLoadError, InputError, SetupError, SupportDecision

KB_DIR = BASE_DIR / "data" / "kb"
P4_PROMPT_DIR = BASE_DIR / "prompts" / "phase4"
EVAL_DIR = BASE_DIR / "eval"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"
RAG_LOG = LOG_DIR / "rag_interactions.jsonl"
CASES_PATH = EVAL_DIR / "rag_cases.yaml"
RETRIEVAL_CONFIG_PATH = RESULTS_DIR / "retrieval_config.json"

EMBEDDING_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "hashing-lexical": 1024}
# Used only until the notebook calibrates a threshold on the evaluation queries (Section 3.5).
DEFAULT_THRESHOLDS = {"text-embedding-3-small": 0.30, "text-embedding-3-large": 0.30, "hashing-lexical": 0.12}


# %% [config]
@dataclass
class RagSettings:
    embedding_model: str = "text-embedding-3-small"
    vector_backend: str = "memory"       # "pinecone" or "memory"
    pinecone_index: str = "support-kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    top_k: int = 4
    threshold_override: Optional[float] = None
    chunking: str = "section"


def load_rag_settings(llm: p3.Settings) -> RagSettings:
    """Embeddings follow the LLM key: no OpenAI key -> offline hashing embeddings (lexical, real but not semantic)."""
    rs = RagSettings(
        embedding_model="hashing-lexical" if llm.dry_run else os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        pinecone_index=os.getenv("PINECONE_INDEX_NAME", "support-kb").strip().lower(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        top_k=int(p3._env_float("RAG_TOP_K", 4)),
        chunking=os.getenv("RAG_CHUNKING", "section").strip(),
    )
    raw_thr = os.getenv("RAG_SCORE_THRESHOLD", "").strip()
    rs.threshold_override = p3._env_float("RAG_SCORE_THRESHOLD", 0.0) if raw_thr else None
    has_pc = bool(os.getenv("PINECONE_API_KEY", "").strip())
    wanted = os.getenv("VECTOR_BACKEND", "auto").strip().lower()
    if wanted == "pinecone" and not has_pc:
        raise SetupError("VECTOR_BACKEND=pinecone but PINECONE_API_KEY is empty in .env.")
    if wanted == "memory" or rs.embedding_model == "hashing-lexical":
        rs.vector_backend = "memory"  # the offline lexical baseline always stays in memory
    else:
        rs.vector_backend = "pinecone" if has_pc else "memory"
    return rs


def resolve_threshold(rs: RagSettings, embedding_name: str, strategy: str) -> Tuple[float, str]:
    """Env override > calibrated value saved by the notebook > default for the embedding model."""
    if rs.threshold_override is not None:
        return rs.threshold_override, "RAG_SCORE_THRESHOLD in .env"
    if RETRIEVAL_CONFIG_PATH.exists():
        try:
            cfg = json.loads(RETRIEVAL_CONFIG_PATH.read_text(encoding="utf-8"))
            if cfg.get("embedding") == embedding_name and cfg.get("chunking") == strategy:
                return float(cfg["threshold"]), "calibrated (results/retrieval_config.json)"
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            pass
    return DEFAULT_THRESHOLDS.get(embedding_name, 0.3), "default (not yet calibrated)"


# %% [corpus]
@dataclass
class KBDocument:
    doc_id: str
    title: str
    version: str
    effective_date: str
    status: str
    category: str
    source: str
    body: str
    path: str


_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)
_REQUIRED_META = {"doc_id", "title", "version", "effective_date", "status", "category"}


def load_corpus(kb_dir: Path = KB_DIR) -> List[KBDocument]:
    """Load every markdown document and validate its metadata.

    Rejects a corpus with two ACTIVE versions of one doc_id: that is a policy
    conflict a human must resolve, not something retrieval should guess about.
    """
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise DataLoadError(f"No .md documents found in {kb_dir}.")
    docs: List[KBDocument] = []
    for path in files:
        m = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
        if not m:
            raise DataLoadError(f"{path.name}: missing '---' front matter block.")
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError as exc:
            raise DataLoadError(f"{path.name}: invalid front matter ({exc}).") from exc
        missing = _REQUIRED_META - set(meta)
        if missing:
            raise DataLoadError(f"{path.name}: front matter missing {sorted(missing)}.")
        if meta["status"] not in {"active", "superseded"}:
            raise DataLoadError(f"{path.name}: status must be active or superseded.")
        body = m.group(2).strip()
        if len(body) < 40:
            raise DataLoadError(f"{path.name}: body is empty or too short.")
        docs.append(KBDocument(doc_id=str(meta["doc_id"]), title=str(meta["title"]), version=str(meta["version"]),
                               effective_date=str(meta["effective_date"]), status=meta["status"],
                               category=str(meta["category"]), source=str(meta.get("source", "official")),
                               body=body, path=path.name))
    active: Dict[str, List[str]] = {}
    for d in docs:
        if d.status == "active":
            active.setdefault(d.doc_id, []).append(d.version)
    clashes = {k: v for k, v in active.items() if len(v) > 1}
    if clashes:
        raise DataLoadError(f"More than one active version: {clashes}. Mark old versions as superseded.")
    return docs


def corpus_fingerprint(docs: List[KBDocument]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.path):
        h.update(f"{d.path}|{d.status}|{d.body}".encode("utf-8"))
    return h.hexdigest()[:8]


# %% [chunking]
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    version: str
    title: str
    section: str
    status: str
    effective_date: str
    category: str
    text: str
    embed_text: str

    @property
    def tokens(self) -> int:
        return approx_tokens(self.text)


def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))  # ~4 characters per token for English


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "part"


def _sections(body: str) -> List[Tuple[str, str]]:
    """Split markdown on '## ' headings. The '# title' line is dropped (title is in metadata)."""
    out: List[Tuple[str, str]] = []
    heading, buf = "Overview", []
    for line in body.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            continue
        if line.startswith("## "):
            if "".join(buf).strip():
                out.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((heading, "\n".join(buf).strip()))
    return out


def _windows(text: str, size: int, overlap: int) -> List[str]:
    """Character windows that end on a word boundary, with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > 0 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        while start < len(text) and start > 0 and text[start - 1] != " ":
            start += 1  # don't start mid-word
    return out


def _make(doc: KBDocument, section: str, text: str, i: int, embed_prefix: str) -> Chunk:
    return Chunk(chunk_id=f"{doc.doc_id}@{doc.version}#{_slug(section)}-{i}", doc_id=doc.doc_id,
                 version=doc.version, title=doc.title, section=section, status=doc.status,
                 effective_date=doc.effective_date, category=doc.category, text=text,
                 embed_text=f"{embed_prefix}\n{text}")


def chunk_whole_doc(doc: KBDocument) -> List[Chunk]:
    text = "\n".join(f"{h}: {t}" for h, t in _sections(doc.body))
    return [_make(doc, "Full document", text, 0, doc.title)]


def chunk_fixed(doc: KBDocument, size: int = 700, overlap: int = 100) -> List[Chunk]:
    text = " ".join(f"{h}. {t}" for h, t in _sections(doc.body))
    return [_make(doc, f"Part {i + 1}", w, i, doc.title) for i, w in enumerate(_windows(text, size, overlap))]


def chunk_section(doc: KBDocument, max_chars: int = 1200, overlap: int = 150) -> List[Chunk]:
    """One chunk per '## ' section, each prefixed with 'Title > Section' so it embeds with its context.
    Long sections are split into overlapping windows."""
    chunks: List[Chunk] = []
    for heading, text in _sections(doc.body):
        for i, w in enumerate(_windows(text, max_chars, overlap)):
            chunks.append(_make(doc, heading, w, i, f"{doc.title} > {heading}"))
    return chunks


CHUNKERS = {"whole_doc": chunk_whole_doc, "fixed_700": chunk_fixed, "section": chunk_section}


def build_chunks(docs: List[KBDocument], strategy: str) -> List[Chunk]:
    if strategy not in CHUNKERS:
        raise SetupError(f"Unknown chunking strategy {strategy!r}. Choose from {sorted(CHUNKERS)}.")
    chunks = [c for d in docs for c in CHUNKERS[strategy](d)]
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise DataLoadError("Duplicate chunk ids; check for repeated section headings.")
    return chunks


# %% [embeddings]
_STOP = set("a an the and or of to in on for with is are was be can do does how what when where which who why "
            "i we you our your my it its this that these those if at by from as not no any all there their them "
            "will would should could just still has have had get".split())


def _tokens(text: str) -> List[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural folding: payouts -> payout
        out.append(t)
    return out


class HashingEmbeddings(Embeddings):
    """Offline LEXICAL baseline: hashed word unigrams + bigrams, L2-normalised.

    Real keyword retrieval with no API, used in dry-run mode and as the
    'keyword search' comparison for semantic embeddings. It cannot match
    synonyms ("sign off" vs "approve"), which is exactly what semantic
    embeddings add.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim)
        for f in feats:
            h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def make_embeddings(name: str, llm: p3.Settings) -> Tuple[Embeddings, int]:
    if name == "hashing-lexical":
        return HashingEmbeddings(EMBEDDING_DIMS[name]), EMBEDDING_DIMS[name]
    if llm.dry_run:
        raise SetupError("OpenAI embeddings need OPENAI_API_KEY (dry-run mode is on).")
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise SetupError("langchain-openai is not installed. Run: pip install -r requirements.txt") from exc
    kwargs: Dict[str, Any] = dict(model=name, max_retries=llm.max_retries, timeout=llm.timeout_s,
                                  check_embedding_ctx_length=False)  # skips a tiktoken download; chunks are short
    if llm.base_url:
        kwargs["base_url"] = llm.base_url
    emb = OpenAIEmbeddings(**kwargs)
    dim = EMBEDDING_DIMS.get(name)
    if dim is None:
        try:
            dim = len(emb.embed_query("dimension probe"))
        except Exception as exc:
            raise SetupError(f"Embedding call failed ({p3.classify_llm_error(exc)}). Check key, base URL, model.") from exc
    return emb, dim


# %% [stores]
@dataclass
class Hit:
    chunk_id: str
    score: float
    metadata: Dict[str, Any]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def chunk_metadata(c: Chunk) -> Dict[str, Any]:
    return {"doc_id": c.doc_id, "version": c.version, "title": c.title, "section": c.section,
            "status": c.status, "effective_date": c.effective_date, "category": c.category, "text": c.text}


class InMemoryVectorStore:
    """Exact cosine search in numpy. Used offline and when no Pinecone key is set."""

    backend = "memory"

    def __init__(self) -> None:
        self._ns: Dict[str, Dict[str, Any]] = {}

    def count(self, namespace: str) -> int:
        return len(self._ns.get(namespace, {}).get("ids", []))

    def upsert(self, namespace: str, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        m = np.array(vectors, dtype=float)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._ns[namespace] = {"ids": list(ids), "matrix": m / norms, "meta": list(metadatas)}

    def query(self, namespace: str, vector: List[float], top_k: int,
              flt: Optional[Dict[str, Any]] = None) -> List[Hit]:
        ns = self._ns.get(namespace)
        if not ns:
            return []
        q = np.array(vector, dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = ns["matrix"] @ q
        order = np.argsort(-scores)
        hits = []
        for i in order:
            meta = ns["meta"][i]
            if flt and any(meta.get(k) != v for k, v in flt.items()):
                continue
            hits.append(Hit(ns["ids"][i], float(scores[i]), meta))
            if len(hits) >= top_k:
                break
        return hits


class PineconeVectorStore:
    """Pinecone serverless index, cosine metric. One namespace per (embedding, chunking, corpus version)."""

    backend = "pinecone"

    def __init__(self, api_key: str, index_name: str, dimension: int, cloud: str, region: str) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise SetupError("The 'pinecone' package is not installed. Run: pip install -r requirements.txt") from exc
        self.pc = Pinecone(api_key=api_key)
        try:
            exists = (self.pc.has_index(index_name) if hasattr(self.pc, "has_index")
                      else index_name in self.pc.list_indexes().names())
        except Exception as exc:
            raise SetupError(f"Pinecone: could not list indexes ({type(exc).__name__}). Check PINECONE_API_KEY.") from exc
        if not exists:
            try:
                self.pc.create_index(name=index_name, dimension=dimension, metric="cosine",
                                     spec=ServerlessSpec(cloud=cloud, region=region))
            except Exception as exc:
                raise SetupError(f"Pinecone: could not create index '{index_name}' in {cloud}/{region} "
                                 f"({type(exc).__name__}: {str(exc)[:160]}). Check PINECONE_CLOUD/PINECONE_REGION "
                                 "and your plan's index limit.") from exc
            self._wait_ready(index_name)
        else:
            dim = _get(self.pc.describe_index(index_name), "dimension")
            if dim and int(dim) != dimension:
                raise SetupError(f"Pinecone index '{index_name}' has dimension {dim}, but the embedding model "
                                 f"produces {dimension}. Set a new PINECONE_INDEX_NAME in .env.")
        self.index = self.pc.Index(name=index_name)
        self.index_name = index_name

    def _wait_ready(self, name: str, timeout_s: int = 120) -> None:
        start = time.time()
        while time.time() - start < timeout_s:
            status = _get(self.pc.describe_index(name), "status")
            if _get(status, "ready", False):
                return
            time.sleep(2)
        raise SetupError(f"Pinecone index '{name}' was not ready after {timeout_s}s.")

    def count(self, namespace: str) -> int:
        stats = self.index.describe_index_stats()
        info = (_get(stats, "namespaces") or {}).get(namespace)
        return int(_get(info, "vector_count", 0) or 0)

    def upsert(self, namespace: str, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        rows = [{"id": i, "values": [float(x) for x in v], "metadata": m} for i, v, m in zip(ids, vectors, metadatas)]
        for b in range(0, len(rows), 100):
            self.index.upsert(vectors=rows[b:b + 100], namespace=namespace)

    def wait_until_count(self, namespace: str, expected: int, timeout_s: int = 90) -> None:
        """Pinecone is eventually consistent: wait until the new vectors are queryable."""
        start = time.time()
        while time.time() - start < timeout_s:
            if self.count(namespace) >= expected:
                return
            time.sleep(2)
        raise SetupError(f"Pinecone namespace '{namespace}' did not reach {expected} vectors in {timeout_s}s.")

    def query(self, namespace: str, vector: List[float], top_k: int,
              flt: Optional[Dict[str, Any]] = None) -> List[Hit]:
        pf = {k: {"$eq": v} for k, v in flt.items()} if flt else None
        res = self.index.query(vector=[float(x) for x in vector], top_k=top_k, namespace=namespace,
                               filter=pf, include_metadata=True)
        return [Hit(str(_get(m, "id")), float(_get(m, "score", 0.0)), dict(_get(m, "metadata") or {}))
                for m in (_get(res, "matches") or [])]


def make_store(rs: RagSettings, dimension: int):
    if rs.vector_backend == "pinecone":
        return PineconeVectorStore(os.getenv("PINECONE_API_KEY", "").strip(), rs.pinecone_index,
                                   dimension, rs.pinecone_cloud, rs.pinecone_region)
    return InMemoryVectorStore()


def namespace_for(embedding_name: str, strategy: str, fingerprint: str) -> str:
    # A new corpus version gets a new namespace, so stale vectors are never mixed in.
    return f"{_slug(embedding_name)}--{strategy}--{fingerprint}"


def build_index(chunks: List[Chunk], embeddings: Embeddings, store: Any, namespace: str) -> Dict[str, Any]:
    """Embed and upsert chunks. Skips work if the namespace already holds them (re-runs cost nothing)."""
    start = time.perf_counter()
    if store.backend == "pinecone" and store.count(namespace) >= len(chunks):
        return {"namespace": namespace, "chunks": len(chunks), "seconds": 0.0, "reused": True}
    try:
        vectors = embeddings.embed_documents([c.embed_text for c in chunks])
    except Exception as exc:
        raise SetupError(f"Embedding failed ({p3.classify_llm_error(exc)}). Check OPENAI_API_KEY / OPENAI_BASE_URL.") from exc
    store.upsert(namespace, [c.chunk_id for c in chunks], vectors, [chunk_metadata(c) for c in chunks])
    if store.backend == "pinecone":
        store.wait_until_count(namespace, len(chunks))
    return {"namespace": namespace, "chunks": len(chunks), "seconds": round(time.perf_counter() - start, 2),
            "reused": False}


# %% [retrieval]
class KBRetriever(BaseRetriever):
    """LangChain retriever over the vector store. LangSmith records each call as a retriever run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    store: Any
    embeddings: Any
    namespace: str
    top_k: int = 4
    active_only: bool = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        hits = self.store.query(self.namespace, vector, self.top_k, {"status": "active"} if self.active_only else None)
        return [Document(page_content=h.metadata.get("text", ""),
                         metadata={**h.metadata, "chunk_id": h.chunk_id, "score": round(h.score, 4)}) for h in hits]


def build_retrieval_query(redacted: str, prior_user_msgs: List[str], account_ctx: List[Dict[str, Any]]) -> str:
    """The text that gets embedded. Adds context the raw message lacks:
    - a short follow-up ("and how long do those take?") is joined to the previous question;
    - lookup facts (return codes, processing method, duplicate flag, error codes) become search terms."""
    parts = [redacted]
    if prior_user_msgs and len(redacted.split()) <= 10:
        parts.insert(0, prior_user_msgs[-1])
    for item in account_ctx:
        data, derived = item.get("data") or {}, item.get("derived") or {}
        if data.get("return_code"):
            parts.append(f"bank return code {data['return_code']}")
        if item["tool"] == "get_payment_status" and data.get("status") == "processing":
            parts.append(f"{data.get('method', '')} settlement time SLA late payment")
        if derived.get("possible_duplicate"):
            parts.append("duplicate charge refund")
        if item["tool"] == "lookup_error_code":
            parts.append(f"sync error {item['id']}")
    return " ".join(parts)


# %% [retrieval_eval]
def load_rag_cases(path: Path = CASES_PATH) -> List[Dict[str, Any]]:
    cases = p3.load_yaml(path) or []
    seen = set()
    for c in cases:
        if "id" not in c or "message" not in c:
            raise DataLoadError(f"Case missing id or message: {c}")
        if c["id"] in seen:
            raise DataLoadError(f"Duplicate case id {c['id']}.")
        seen.add(c["id"])
        c.setdefault("retrieval_only", False)
        c.setdefault("category", "other")
        c.setdefault("split", "phase4")
        c.setdefault("safety_critical", False)
        c.setdefault("must_include", [])
        c.setdefault("must_exclude", [])
        c.setdefault("prior_turns", [])
        c.setdefault("relevant_docs", None)
        if not c["retrieval_only"]:
            routes = c.get("expected_routes")
            if not routes or set(routes) - set(p3.ROUTES):
                raise DataLoadError(f"{c['id']}: expected_routes missing or invalid.")
    return cases


def evaluate_retrieval(cases: List[Dict[str, Any]], embeddings: Embeddings, store: Any, namespace: str,
                       tools: p3.MockTools, chunk_tokens: Dict[str, int], k_max: int = 5,
                       active_only: bool = True):
    """Per-query retrieval metrics. Relevance is judged at document level (any chunk of a relevant doc counts)."""
    import pandas as pd

    rows = []
    for c in cases:
        if c["relevant_docs"] is None:
            continue
        redacted = p3.redact_pii(c["message"])[0]
        ctx = p3.gather_account_context(tools, p3.extract_entities(redacted))
        prior = [p3.redact_pii(p)[0] for p in c["prior_turns"]]
        query = build_retrieval_query(redacted, prior, ctx)
        hits = store.query(namespace, embeddings.embed_query(query), k_max,
                           {"status": "active"} if active_only else None)
        docs = [h.metadata.get("doc_id") for h in hits]
        rel = set(c["relevant_docs"])
        first = next((i + 1 for i, d in enumerate(docs) if d in rel), None)
        row = {"id": c["id"], "category": c["category"], "answerable": bool(rel), "relevant": sorted(rel),
               "retrieved": docs, "first_relevant_rank": first,
               "top1_score": round(hits[0].score, 4) if hits else 0.0,
               "mrr": (1.0 / first) if first else 0.0,
               "recall@4": (len(rel & set(docs[:4])) / len(rel)) if rel else None,
               "stale_in_top5": any(h.metadata.get("status") != "active" for h in hits),
               "context_tokens@4": sum(chunk_tokens.get(h.chunk_id, 0) for h in hits[:4])}
        for k in (1, 3, 4, 5):
            row[f"hit@{k}"] = bool(first and first <= k)
        rows.append(row)
    return pd.DataFrame(rows)


def retrieval_summary(per_query) -> Dict[str, Any]:
    a = per_query[per_query.answerable]
    return {"queries": int(len(a)), "hit@1": round(a["hit@1"].mean(), 3), "hit@3": round(a["hit@3"].mean(), 3),
            "hit@4": round(a["hit@4"].mean(), 3), "hit@5": round(a["hit@5"].mean(), 3),
            "mrr": round(a["mrr"].mean(), 3), "recall@4": round(a["recall@4"].mean(), 3),
            "avg_context_tokens@4": round(per_query["context_tokens@4"].mean(), 0),
            "stale_leaks": int(per_query["stale_in_top5"].sum())}


def calibrate_threshold(per_query, min_answerable_kept: float = 0.95) -> Tuple[float, Dict[str, Any]]:
    """Choose the top-1 relevance cut-off below which retrieval counts as 'nothing found'.

    Rule: the HIGHEST threshold that still keeps at least `min_answerable_kept` of the
    answerable queries. The threshold is only the first, cheap filter: blocking a real
    answer always costs the customer an escalation, while a weak excerpt that slips through
    is still checked by the prompt rules (Step 4) and the number verifier. So recall of
    real answers is protected first, and whatever unanswerable queries it blocks is a bonus.
    """
    pos = np.sort(per_query[per_query.answerable].top1_score.to_numpy())
    neg = per_query[~per_query.answerable].top1_score.to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Calibration needs both answerable and unanswerable queries.")
    allowed_drops = int(np.floor(len(pos) * (1 - min_answerable_kept)))
    t = float(pos[allowed_drops])  # keeps pos[allowed_drops:], i.e. >= min_answerable_kept of answerable
    t = round(t - 1e-4, 4)         # just below that score so the query itself is kept
    return t, {"min_answerable_kept": min_answerable_kept,
               "answerable_kept": round(float((pos >= t).mean()), 3),
               "unanswerable_blocked": round(float((neg < t).mean()), 3),
               "answerable_n": int(len(pos)), "unanswerable_n": int(len(neg)),
               "answerable_score_range": [round(float(pos.min()), 3), round(float(pos.max()), 3)],
               "unanswerable_score_range": [round(float(neg.min()), 3), round(float(neg.max()), 3)]}


# %% [prompts]
def load_p4_prompts(prompt_dir: Path = P4_PROMPT_DIR) -> Dict[str, p3.PromptVersion]:
    registry: Dict[str, p3.PromptVersion] = {}
    for path in sorted(prompt_dir.glob("v*.yaml")):
        raw = p3.load_yaml(path) or {}
        missing = {"id", "created", "description", "changelog", "system"} - set(raw)
        if missing:
            raise DataLoadError(f"{path.name} is missing {sorted(missing)}.")
        pv = p3.PromptVersion(id=raw["id"], parent=raw.get("parent"), created=str(raw["created"]),
                              description=raw["description"], changelog=list(raw["changelog"]),
                              uses_fewshot=bool(raw.get("uses_fewshot")), system=raw["system"])
        if pv.uses_fewshot and "{{FEWSHOT}}" not in pv.system:
            raise DataLoadError(f"{path.name}: uses_fewshot is true but {{{{FEWSHOT}}}} is missing.")
        pv.sha = hashlib.sha256(pv.system.encode("utf-8")).hexdigest()[:10]
        registry[pv.id] = pv
    if not registry:
        raise DataLoadError(f"No prompt files in {prompt_dir}.")
    return registry


def load_rag_fewshot(prompt_dir: Path = P4_PROMPT_DIR) -> List[Dict[str, Any]]:
    examples = p3.load_yaml(prompt_dir / "fewshot_rag.yaml") or []
    for ex in examples:
        SupportDecision(**ex["decision"])  # must match the output schema
    return examples


def render_p4_system(pv: p3.PromptVersion, fewshot: List[Dict[str, Any]]) -> str:
    text = pv.system.replace("{{TODAY}}", p3.REFERENCE_DATE.isoformat())
    if pv.uses_fewshot:
        text = text.replace("{{FEWSHOT}}", p3.format_fewshot(fewshot))
    return text


# %% [agent]
@dataclass
class Excerpt:
    chunk_id: str
    doc_id: str
    version: str
    title: str
    section: str
    text: str
    score: Optional[float]
    source: str  # "retrieved", "direct_reference" or "full_kb"


def excerpt_from_chunk(c: Chunk, source: str, score: Optional[float] = None) -> Excerpt:
    return Excerpt(c.chunk_id, c.doc_id, c.version, c.title, c.section, c.text, score, source)


def format_excerpts(excerpts: List[Excerpt], mode: str, note: str = "") -> str:
    if not excerpts:
        return f"POLICY EXCERPTS:\nNONE ({note or 'no approved document matched this question'})"
    header = ("POLICY EXCERPTS (entire knowledge base, active versions):" if mode == "full"
              else "POLICY EXCERPTS (retrieved; approved, active versions only):")
    blocks = []
    for i, e in enumerate(excerpts, 1):
        tag = f"relevance {e.score:.2f}" if e.score is not None else e.source.replace("_", " ")
        blocks.append(f"[E{i}] {e.doc_id} v{e.version} | {e.title} > {e.section} | {tag}\n{e.text}")
    return header + "\n" + "\n\n".join(blocks)


def verify_rag_decision(d: SupportDecision, excerpts: List[Excerpt], account_ctx: List[Dict[str, Any]],
                        user_text: str) -> Tuple[List[str], List[str]]:
    """Phase 3 checks, tightened for RAG: citations must be docs that were actually IN the context,
    and every number must appear in a cited excerpt, the lookups or the user's message."""
    issues: List[str] = []
    in_context = {e.doc_id for e in excerpts}
    if any(c not in in_context for c in d.citations):
        issues.append("invalid_citation")
    cited = " ".join(f"{e.text} {e.version} {e.title} {e.section}" for e in excerpts if e.doc_id in d.citations)
    allowed = f"{cited} {json.dumps(account_ctx)} {user_text} {p3.REFERENCE_DATE.isoformat()}"
    unsupported = sorted(p3._numbers(p3._LIST_MARKER.sub(" ", d.answer)) - p3._numbers(allowed))
    if unsupported:
        issues.append("unsupported_number")
    if p3.CANARY.lower() in d.answer.lower():
        issues.append("prompt_leak")
    for m in p3._CREDENTIAL_ASK.finditer(d.answer):
        if not p3._NEGATION.search(d.answer[max(0, m.start() - 30):m.start() + len(m.group(1))]):
            issues.append("credential_request")
            break
    if d.route == "escalated" and d.escalation is None:
        issues.append("missing_escalation")
    return issues, unsupported


@dataclass
class RagTurnResult(p3.TurnResult):
    context_mode: str = "rag"
    retrieval_query: str = ""
    excerpt_chunk_ids: List[str] = field(default_factory=list)
    excerpt_doc_ids: List[str] = field(default_factory=list)
    top_score: Optional[float] = None
    retrieval_empty: Optional[bool] = None
    excerpt_tokens: int = 0
    partial_upgrade: bool = False


class RAGSupportAgent:
    """Phase 3 agent + knowledge context. mode: 'none' (no documents), 'full' (whole KB) or 'rag'."""

    def __init__(self, active_chunks: List[Chunk], retriever: Optional[KBRetriever], tools: p3.MockTools,
                 llm: p3.StructuredLLM, prompt: p3.PromptVersion, fewshot: List[Dict[str, Any]],
                 mode: str = "rag", threshold: float = 0.3, use_threshold: bool = True,
                 excerpt_block: bool = True, logger: Any = None, log_path: Optional[Path] = RAG_LOG) -> None:
        if mode not in {"none", "full", "rag"}:
            raise ValueError("mode must be none, full or rag")
        if mode == "rag" and retriever is None:
            raise ValueError("rag mode needs a retriever")
        self.active_chunks = active_chunks
        self.by_doc: Dict[str, List[Chunk]] = {}
        for c in active_chunks:
            self.by_doc.setdefault(c.doc_id, []).append(c)
        self.retriever, self.tools, self.llm, self.prompt = retriever, tools, llm, prompt
        self.system_prompt = render_p4_system(prompt, fewshot)
        self.mode, self.threshold, self.use_threshold, self.excerpt_block = mode, threshold, use_threshold, excerpt_block
        self.logger = logger or p3.setup_logging()
        self.log_path = log_path
        self.history: List[BaseMessage] = []
        self.prior_user_msgs: List[str] = []
        self.session_id = uuid.uuid4().hex[:12]
        self.turn = 0

    # ----- knowledge context -----
    def _context(self, redacted: str, account_ctx: List[Dict[str, Any]], run_config: Dict[str, Any]):
        if self.mode == "none":
            return [], "", None, None, "no documents provided"
        if self.mode == "full":
            return [excerpt_from_chunk(c, "full_kb") for c in self.active_chunks], "", None, False, ""
        query = build_retrieval_query(redacted, self.prior_user_msgs, account_ctx)
        docs = self.retriever.invoke(query, config={**run_config, "run_name": "kb_retrieval"})
        top = docs[0].metadata["score"] if docs else None
        kept = [d for d in docs if not self.use_threshold or d.metadata["score"] >= self.threshold]
        excerpts = [Excerpt(d.metadata["chunk_id"], d.metadata["doc_id"], str(d.metadata["version"]),
                            d.metadata["title"], d.metadata["section"], d.page_content, d.metadata["score"],
                            "retrieved") for d in kept]
        # Direct document reference: an error-code lookup names its KB article, so include it even if
        # semantic search ranked it low. Deterministic, and it never depends on wording.
        present = {e.chunk_id for e in excerpts}
        for item in account_ctx:
            doc_id = (item.get("data") or {}).get("kb_doc_id")
            if item["tool"] == "lookup_error_code" and doc_id in self.by_doc:
                excerpts += [excerpt_from_chunk(c, "direct_reference") for c in self.by_doc[doc_id]
                             if c.chunk_id not in present]
        empty = not excerpts
        note = (f"no approved document matched this question; best relevance {top:.2f} is below {self.threshold:.2f}"
                if empty and top is not None else "")
        return excerpts, query, top, empty, note

    # ----- one turn -----
    def respond(self, message: Any, extra_metadata: Optional[Dict[str, Any]] = None) -> RagTurnResult:
        start = time.perf_counter()
        self.turn += 1
        base = dict(raw_route=None, intents=[], citations=[], unanswered=[], ticket_id=None, escalation_team=None,
                    verifier_issues=[], unsupported_numbers=[], guardrail_fired=False, error=None,
                    prompt_version=self.prompt.id, prompt_sha=self.prompt.sha, model=self.llm.model_name,
                    latency_ms=0.0, input_tokens=0, output_tokens=0, pii_redactions={}, decision_notes="",
                    context_mode=self.mode)
        try:
            text = p3.LLMSupportAgent._validate(message)
        except InputError as exc:
            result = RagTurnResult(answer=str(exc), route="needs_info", **{**base, "intents": ["invalid_input"]})
            self._log(message, result)
            return result

        redacted, pii = p3.redact_pii(text)
        account_ctx = p3.gather_account_context(self.tools, p3.extract_entities(redacted))
        run_config = {"tags": ["phase4", self.mode, self.prompt.id],
                      "metadata": {"prompt_version": self.prompt.id, "prompt_sha": self.prompt.sha,
                                   "context_mode": self.mode, "session_id": self.session_id, "turn": self.turn,
                                   **(extra_metadata or {})}}
        try:
            excerpts, query, top, empty, note = self._context(redacted, account_ctx, run_config)
        except Exception as exc:  # vector store or embedding outage: answer without documents, safely
            self.logger.error("retrieval_failure session=%s type=%s", self.session_id, type(exc).__name__)
            excerpts, query, top, empty, note = [], "", None, True, "knowledge base temporarily unavailable"

        user_turn = p3.build_user_turn(redacted, account_ctx)
        if self.excerpt_block:
            user_turn += "\n\n" + format_excerpts(excerpts, self.mode, note)
        messages = [SystemMessage(self.system_prompt), *self.history[-p3.MAX_HISTORY_MESSAGES:], HumanMessage(user_turn)]
        llm_res = self.llm.invoke(messages, {**run_config, "run_name": f"support_decision:{self.mode}"})

        ctx_fields = dict(retrieval_query=p3.redact_pii(query)[0], excerpt_chunk_ids=[e.chunk_id for e in excerpts],
                          excerpt_doc_ids=sorted({e.doc_id for e in excerpts}), top_score=top,
                          retrieval_empty=empty, excerpt_tokens=sum(approx_tokens(e.text) for e in excerpts))
        if llm_res.decision is None:
            ticket = self.tools.create_ticket("general", "high", f"Agent could not answer ({llm_res.error}).")
            result = RagTurnResult(
                answer=f"Sorry, I couldn't process that right now. I've passed it to a human agent (ticket {ticket}).",
                route="escalated", **{**base, "ticket_id": ticket, "escalation_team": "general",
                                      "error": llm_res.error, "input_tokens": llm_res.input_tokens,
                                      "output_tokens": llm_res.output_tokens, "pii_redactions": pii}, **ctx_fields)
        else:
            d = llm_res.decision
            issues, unsupported = verify_rag_decision(d, excerpts, account_ctx, redacted)
            blocking = [i for i in issues if i in p3.BLOCKING_ISSUES]
            route, answer = d.route, d.answer.strip()
            team = d.escalation.team if d.escalation else None
            ticket, partial = None, False
            if blocking:
                ticket = self.tools.create_ticket("general", "high", f"Guardrail blocked an answer: {', '.join(blocking)}.")
                route, team, answer = "escalated", "general", f"{p3.SAFE_FALLBACK} (ticket {ticket})"
            else:
                if d.unanswered and route == "resolved":  # backstop for missing information
                    route, team, partial = "escalated", team or "general", True
                if route == "escalated" or (route == "refused" and d.escalation is not None):
                    team = team or "general"
                    priority = d.escalation.priority if d.escalation else "normal"
                    reason = d.escalation.reason if d.escalation else f"Unanswered: {', '.join(d.unanswered)}."
                    ticket = self.tools.create_ticket(team, priority, f"Intents: {', '.join(d.intents)}. {reason}")
                    answer = f"{answer}\n\nReference: ticket {ticket}."
            result = RagTurnResult(
                answer=answer, route=route,
                **{**base, "raw_route": d.route, "intents": d.intents, "citations": d.citations,
                   "unanswered": d.unanswered, "ticket_id": ticket, "escalation_team": team,
                   "verifier_issues": issues, "unsupported_numbers": unsupported, "guardrail_fired": bool(blocking),
                   "input_tokens": llm_res.input_tokens, "output_tokens": llm_res.output_tokens,
                   "pii_redactions": pii, "decision_notes": d.decision_notes},
                **ctx_fields, partial_upgrade=partial)

        result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        self.history += [HumanMessage(redacted), AIMessage(result.answer)]
        self.prior_user_msgs.append(redacted)
        self._log(message, result)
        return result

    def _log(self, message: Any, result: RagTurnResult) -> None:
        if self.log_path is None:
            return
        raw = message if isinstance(message, str) else repr(message)
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session_id": self.session_id,
                  "turn": self.turn, "user_message_redacted": p3.redact_pii(raw[:p3.MAX_INPUT_CHARS])[0],
                  **{k: v for k, v in asdict(result).items() if k not in {"answer", "decision_notes"}},
                  "answer_redacted": p3.redact_pii(result.answer)[0]}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            self.logger.warning("log_write_failed type=%s", type(exc).__name__)


# %% [experiments]
CONDITIONS: Dict[str, Dict[str, Any]] = {
    "no_docs_open": {"prompt": "v5_open_control", "mode": "none", "excerpt_block": False},
    "no_docs_strict": {"prompt": "v5_rag_grounded", "mode": "none", "excerpt_block": True},
    "full_context": {"prompt": "v5_rag_grounded", "mode": "full"},
    "rag": {"prompt": "v5_rag_grounded", "mode": "rag"},
}
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "rag_no_threshold": {"prompt": "v5_rag_grounded", "mode": "rag", "use_threshold": False,
                         "categories": ["missing_info", "partial"]},
    "rag_no_version_filter": {"prompt": "v5_rag_grounded", "mode": "rag", "active_only": False,
                              "case_ids": ["T1_gbp_fee_current_version", "M1_same_day_fee_and_cutoff",
                                           "A12_failed_payment_fee", "PA1_card_fee_and_mobile_app"]},
}


@dataclass
class RagComponents:
    """Everything an experiment needs, built once."""
    settings: p3.Settings
    rag_settings: RagSettings
    embeddings: Embeddings
    embedding_name: str
    store: Any
    namespace: str
    threshold: float
    active_chunks: List[Chunk]
    prompts: Dict[str, p3.PromptVersion]
    fewshot: List[Dict[str, Any]]
    db: Dict[str, Any]


def make_agent(comp: RagComponents, spec: Dict[str, Any], log_path: Optional[Path] = RAG_LOG) -> RAGSupportAgent:
    retriever = KBRetriever(store=comp.store, embeddings=comp.embeddings, namespace=comp.namespace,
                            top_k=comp.rag_settings.top_k, active_only=spec.get("active_only", True))
    llm = p3.StructuredLLM(p3.make_chat_model(comp.settings), comp.settings)
    return RAGSupportAgent(comp.active_chunks, retriever, p3.MockTools(comp.db, "T-001"), llm,
                           comp.prompts[spec["prompt"]], comp.fewshot, mode=spec["mode"], threshold=comp.threshold,
                           use_threshold=spec.get("use_threshold", True), excerpt_block=spec.get("excerpt_block", True),
                           log_path=log_path)


def run_rag_experiment(conditions: Dict[str, Dict[str, Any]], cases: List[Dict[str, Any]], comp: RagComponents,
                       results_path: Path, resume: bool = True, pause_s: float = 0.0) -> List[Dict[str, Any]]:
    """Every case x every condition, one scored row each, appended as it's produced (resumable)."""
    llm_cases = [c for c in cases if not c["retrieval_only"]]
    done = {(r["variant"], r["case_id"]) for r in p3.read_jsonl(results_path)} if resume else set()
    todo = []
    for name, spec in conditions.items():
        subset = [c for c in llm_cases
                  if ("categories" not in spec or c["category"] in spec["categories"])
                  and ("case_ids" not in spec or c["id"] in spec["case_ids"])]
        todo += [(name, spec, c) for c in subset if (name, c["id"]) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {results_path.name}).")
    for i, (name, spec, case) in enumerate(todo, 1):
        agent = make_agent(comp, spec)
        meta = {"case_id": case["id"], "experiment": "phase4_retrieval", "condition": name}
        for prior in case["prior_turns"]:
            agent.respond(prior, extra_metadata=meta)
        res = agent.respond(case["message"], extra_metadata=meta)
        rel = case["relevant_docs"]
        row = {"variant": name, "case_id": case["id"], "category": case["category"], "split": case["split"],
               "safety_critical": case["safety_critical"], "dry_run": comp.settings.dry_run,
               "expected_routes": case["expected_routes"], "relevant_docs": rel,
               **p3.score(case, res.answer, res.route, res.error, res.guardrail_fired),
               "raw_route_ok": (res.raw_route in case["expected_routes"]) if res.raw_route
               else (res.error is None and res.route in case["expected_routes"]),
               "retrieval_hit": (bool(set(rel) & set(res.excerpt_doc_ids)) if rel else None) if spec["mode"] != "none" else None,
               "citations_relevant": (set(res.citations) <= set(rel)) if (rel and res.citations) else None,
               **{k: v for k, v in asdict(res).items() if k != "pii_redactions"},
               "answer": p3.redact_pii(res.answer)[0], "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        p3.append_jsonl(results_path, row)
        print(f"[{i}/{len(todo)}] {name:<22} {case['id']:<34} {res.route:<10} "
              f"{'PASS' if row['passed'] else 'FAIL ' + str(row['failure_type'])}")
        if res.error in p3.FATAL_ERRORS:
            raise SetupError(f"Stopping: {res.error}. Check OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL.")
        if pause_s:
            time.sleep(pause_s)
    return p3.read_jsonl(results_path)


def summarise_conditions(rows: List[Dict[str, Any]], settings: p3.Settings,
                         cases: Optional[List[Dict[str, Any]]] = None):
    import pandas as pd

    df = pd.DataFrame(rows)
    if cases is not None:  # baseline rows (from Phase 3's run_baseline) don't carry relevant_docs
        rel = {c["id"]: c["relevant_docs"] for c in cases}
        df["relevant_docs"] = df.case_id.map(rel)
    out = []
    for v, g in df.groupby("variant", sort=False):
        missing = g[g.category.isin(["missing_info", "partial"])]
        answerable = g[g.relevant_docs.apply(lambda r: isinstance(r, list) and len(r) > 0)]
        out.append({
            "condition": v, "cases": len(g),
            "pass_rate": round(g.passed.mean(), 3),
            "answerable_pass": round(answerable.passed.mean(), 3) if len(answerable) else None,
            "missing_info_pass": round(missing.passed.mean(), 3) if len(missing) else None,
            "safety_fails": int((g.safety_critical & ~g.passed).sum()),
            "fabrications_caught": int(g.unsupported_numbers.apply(lambda x: bool(x) if isinstance(x, list) else False).sum()),
            "guardrail_fired": int(g.guardrail_fired.sum()),
            "errors": int(g.error.notna().sum()),
            "avg_excerpt_tokens": round(g["excerpt_tokens"].mean(), 0) if "excerpt_tokens" in g else 0,
            "avg_in_tokens": round(g.input_tokens.mean(), 0),
            "usd_per_1k_turns": round((g.input_tokens.mean() * settings.price_input_per_1m
                                       + g.output_tokens.mean() * settings.price_output_per_1m) / 1000, 3),
            "avg_latency_ms": round(g.latency_ms.mean(), 0),
        })
    return pd.DataFrame(out)
