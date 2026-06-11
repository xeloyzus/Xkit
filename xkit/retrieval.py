from __future__ import annotations

import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .logging import get_logger

if TYPE_CHECKING:
    pass

_logger = get_logger(__name__)

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical retrieval.

    Splits identifiers on snake_case and camelCase so that a query for
    "stripe webhook" matches `handleStripeWebhook` and `stripe_webhook_handler`.
    The original compound token is kept as well, so exact-identifier queries
    still rank highest.
    """
    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        lower = word.lower()
        tokens.append(lower)
        # split snake_case
        parts = [p for p in lower.split("_") if p]
        # split camelCase on the original casing
        camel_parts = [p.lower() for p in CAMEL_RE.split(word) if p]
        for p in parts + camel_parts:
            if p != lower:
                tokens.append(p)
    return tokens


@dataclass
class RetrievalResult:
    chunk: dict
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]: ...


def _chunk_text_for_indexing(chunk: dict) -> str:
    return (
        chunk.get("text", "")
        + " "
        + chunk.get("file", "")
        + " "
        + str(chunk.get("symbol") or "")
    )


class BM25Retriever:
    """Dependency-free lexical retrieval using Okapi BM25.

    BM25 outperforms plain TF-IDF cosine for code search because its document
    length normalization handles the highly variable chunk sizes produced by
    symbol-aware chunking, and its saturating term frequency prevents long
    chunks with many repeats from dominating.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(_chunk_text_for_indexing(c)) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.n = len(chunks)

        self.df: dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for tok in set(toks):
                self.df[tok] += 1

        # Inverted index: token -> list[(doc_idx, term_freq)]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, toks in enumerate(self.doc_tokens):
            for tok, tf in Counter(toks).items():
                self.postings[tok].append((idx, tf))

    def _idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []
        scores: dict[int, float] = defaultdict(float)
        for tok in set(q_tokens):
            idf = self._idf(tok)
            for doc_idx, tf in self.postings.get(tok, ()):  # only docs containing tok
                dl = self.doc_lens[doc_idx] or 1
                denom = tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                scores[doc_idx] += idf * (tf * (self.K1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [RetrievalResult(self.chunks[i], s) for i, s in ranked if s > 0]


class LocalTfidfRetriever:
    """Legacy TF-IDF cosine retriever. Kept for backward compatibility;
    prefer :class:`BM25Retriever` (``retriever = "bm25"``)."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(_chunk_text_for_indexing(c)) for c in chunks]
        self.df: dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for tok in set(toks):
                self.df[tok] += 1
        self.n = max(1, len(chunks))
        self.doc_vecs = [self._tfidf(Counter(toks)) for toks in self.doc_tokens]
        self.doc_norms = [
            math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in self.doc_vecs
        ]

    def _idf(self, token: str) -> float:
        return math.log((self.n + 1) / (self.df.get(token, 0) + 1)) + 1.0

    def _tfidf(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {tok: (cnt / total) * self._idf(tok) for tok, cnt in counts.items()}

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        qvec = self._tfidf(Counter(tokenize(query)))
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        results: list[RetrievalResult] = []
        for idx, dvec in enumerate(self.doc_vecs):
            score = 0.0
            for tok, qv in qvec.items():
                score += qv * dvec.get(tok, 0.0)
            score /= qnorm * self.doc_norms[idx]
            if score > 0:
                results.append(RetrievalResult(self.chunks[idx], score))
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Embedding-based retrieval
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_sentence_transformer(model_name: str, device: str):
    """Load (and cache) a sentence-transformers model once per process."""
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def embedding_input_text(chunk: dict) -> str:
    """Canonical text used when embedding a chunk (index & query side must match)."""
    text = chunk.get("text", "") + "\nFile: " + chunk.get("file", "")
    if chunk.get("symbol"):
        text += "\nSymbol: " + str(chunk.get("symbol", ""))
    return text


class FaissQueryRetriever:
    """Query the FAISS index persisted at index time.

    Only the *query string* is encoded per search — the corpus embeddings are
    read from disk. This is the production path: O(1) model calls per query
    instead of re-embedding the entire repo.
    """

    def __init__(
        self,
        chunks: list[dict],
        persist_dir: Path,
        namespace: str,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        from .embeddings import FAISSEmbeddingStore  # raises ImportError without faiss

        self._store = FAISSEmbeddingStore(str(persist_dir))
        self._namespace = namespace
        self._model = _get_sentence_transformer(model_name, device)
        self._by_id = {c.get("chunk_id"): c for c in chunks}
        if not self._store.has_namespace(namespace):
            raise FileNotFoundError(
                f"No persisted FAISS index for namespace {namespace!r} in {persist_dir}"
            )

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        qvec = self._model.encode([query], normalize_embeddings=True)
        rows = self._store.query(self._namespace, qvec, top_k=top_k)
        results: list[RetrievalResult] = []
        for hit in rows[0] if rows else []:
            chunk = self._by_id.get(hit["id"])
            if chunk is not None and hit["score"] > 0:
                results.append(RetrievalResult(chunk, hit["score"]))
        return results


class EmbeddingRetriever:
    """In-memory semantic search: encodes the whole corpus on construction.

    Only suitable for small repos or one-off use — prefer
    :class:`FaissQueryRetriever`, which reads the persisted index.
    """

    def __init__(
        self,
        chunks: list[dict],
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        import numpy  # noqa: F401  (validate dependency early)

        self.chunks = chunks
        self._model = _get_sentence_transformer(model_name, device)
        self._embeddings = None
        texts = [embedding_input_text(c) for c in chunks]
        if texts:
            self._embeddings = self._model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            )

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        import numpy as np

        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        qvec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, qvec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalResult(self.chunks[i], float(scores[i]))
            for i in top_indices
            if float(scores[i]) > 0
        ]


class HybridRetriever:
    """Fuse lexical (BM25) and semantic rankings with Reciprocal Rank Fusion.

    RRF(d) = sum over rankers of 1 / (k + rank_d). Lexical retrieval nails
    exact identifiers ("fix handleStripeWebhook"); embeddings catch conceptual
    queries ("where do we validate payment events"). Fusing both is the
    strongest default for code search.
    """

    RRF_K = 60

    def __init__(self, lexical: Retriever, semantic: Retriever):
        self._lexical = lexical
        self._semantic = semantic

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        pool = max(top_k * 3, 20)
        fused: dict[str, float] = defaultdict(float)
        chunk_by_id: dict[str, dict] = {}
        for retriever in (self._lexical, self._semantic):
            try:
                ranked = retriever.search(query, top_k=pool)
            except Exception as e:  # semantic side may fail at runtime
                _logger.warning("Hybrid sub-retriever failed: %s", e)
                continue
            for rank, result in enumerate(ranked):
                cid = result.chunk.get("chunk_id") or id(result.chunk)
                fused[cid] += 1.0 / (self.RRF_K + rank + 1)
                chunk_by_id[cid] = result.chunk
        ranked_ids = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [RetrievalResult(chunk_by_id[cid], score) for cid, score in ranked_ids]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_semantic(
    chunks: list[dict],
    model_name: str,
    device: str,
    project_root: Path | None,
    index_dir_name: str,
) -> Retriever:
    """Best available semantic retriever: persisted FAISS first, in-memory second."""
    if project_root is not None:
        try:
            return FaissQueryRetriever(
                chunks,
                persist_dir=Path(project_root) / index_dir_name / "faiss",
                namespace=Path(project_root).name,
                model_name=model_name,
                device=device,
            )
        except Exception as e:
            _logger.info("Persisted FAISS index unavailable (%s); using in-memory embeddings", e)
    return EmbeddingRetriever(chunks, model_name=model_name, device=device)


def create_retriever(
    chunks: list[dict],
    method: str = "bm25",
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    project_root: Path | None = None,
    index_dir_name: str = ".xkit",
) -> Retriever:
    """Factory: returns the configured retriever, degrading gracefully.

    Methods:
      - ``bm25`` (default): dependency-free lexical search
      - ``tfidf``: legacy lexical search
      - ``embeddings``: semantic search (persisted FAISS if available)
      - ``hybrid``: BM25 + semantic fused with RRF (recommended when
        sentence-transformers is installed)
    """
    if method == "tfidf":
        return LocalTfidfRetriever(chunks)
    if method == "embeddings":
        try:
            return _build_semantic(chunks, model_name, device, project_root, index_dir_name)
        except Exception as e:
            warnings.warn(
                f"Embedding retriever failed to initialize: {e} — falling back to BM25",
                stacklevel=2,
            )
            return BM25Retriever(chunks)
    if method == "hybrid":
        lexical = BM25Retriever(chunks)
        try:
            semantic = _build_semantic(chunks, model_name, device, project_root, index_dir_name)
            return HybridRetriever(lexical, semantic)
        except Exception as e:
            warnings.warn(
                f"Hybrid semantic side unavailable: {e} — using BM25 only", stacklevel=2
            )
            return lexical
    return BM25Retriever(chunks)
