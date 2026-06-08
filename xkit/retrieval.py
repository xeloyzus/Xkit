from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


@dataclass
class RetrievalResult:
    chunk: dict
    score: float


class LocalTfidfRetriever:
    """Dependency-free local retrieval using TF-IDF cosine similarity.

    This is a practical stand-in for vector embeddings. It behaves like a cheap local index and can
    be replaced by OpenAI embeddings, FAISS, Qdrant, Chroma, etc.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.doc_tokens = [
            tokenize(c.get("text", "") + " " + c.get("file", "") + " " + str(c.get("symbol") or ""))
            for c in chunks
        ]
        self.df: dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for tok in set(toks):
                self.df[tok] += 1
        self.n = max(1, len(chunks))
        self.doc_vecs = [self._tfidf(Counter(toks)) for toks in self.doc_tokens]
        self.doc_norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in self.doc_vecs]

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


class EmbeddingRetriever:
    """Semantic search using sentence-transformers embeddings.

    Requires: pip install sentence-transformers numpy
    Falls back gracefully if the package is not installed.
    """

    def __init__(self, chunks: list[dict], model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.chunks = chunks
        self.model_name = model_name
        self.device = device
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._build()

    def _lazy_import(self):
        """Import heavy dependencies lazily so the module loads fast."""
        global np
        try:
            import numpy as np_impl
            global np
            np = np_impl
        except ImportError:
            raise ImportError(
                "numpy is required for EmbeddingRetriever. Install: pip install numpy"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingRetriever. "
                "Install: pip install sentence-transformers"
            )
        return SentenceTransformer

    def _build(self):
        SentenceTransformer = self._lazy_import()
        self._model = SentenceTransformer(self.model_name, device=self.device)
        texts = [
            c.get("text", "") + "\nFile: " + c.get("file", "") + ("\nSymbol: " + str(c.get("symbol", "")) if c.get("symbol") else "")
            for c in self.chunks
        ]
        if texts:
            self._embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            self._embeddings = np.array([], dtype=np.float32)

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        if self._model is None or self._embeddings is None or len(self._embeddings) == 0:
            return []
        qvec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, qvec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append(RetrievalResult(self.chunks[idx], score))
        return results


def create_retriever(
    chunks: list[dict],
    method: str = "tfidf",
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
):
    """Factory: returns the appropriate retriever based on config."""
    if method == "embeddings":
        try:
            return EmbeddingRetriever(chunks, model_name=model_name, device=device)
        except Exception as e:
            # Any error while initializing embeddings (missing deps, incompatible
            # versions, runtime errors) should gracefully fall back to TF-IDF.
            import warnings
            warnings.warn(f"Embedding retriever failed to initialize: {e} — falling back to TF-IDF retriever")
            return LocalTfidfRetriever(chunks)
    return LocalTfidfRetriever(chunks)
