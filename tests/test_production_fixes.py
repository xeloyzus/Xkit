"""Tests for the production fixes: BM25, tokenization, hybrid fusion, budget."""
from __future__ import annotations

from pathlib import Path

from xkit.agent_context import retrieve_context
from xkit.config import XkitConfig
from xkit.indexer import build_index
from xkit.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RetrievalResult,
    tokenize,
)


def _chunk(cid: str, file: str, text: str, symbol: str | None = None) -> dict:
    return {
        "chunk_id": cid,
        "file": file,
        "kind": "symbol" if symbol else "block",
        "symbol": symbol,
        "start_line": 1,
        "end_line": 10,
        "text": text,
        "hash": cid,
        "token_estimate": max(1, len(text) // 4),
    }


# --- tokenize ---------------------------------------------------------------

def test_tokenize_splits_camel_case():
    toks = tokenize("handleStripeWebhook")
    assert "handlestripewebhook" in toks
    assert "stripe" in toks
    assert "webhook" in toks


def test_tokenize_splits_snake_case():
    toks = tokenize("stripe_webhook_handler")
    assert "stripe" in toks and "webhook" in toks and "handler" in toks


# --- BM25 -------------------------------------------------------------------

def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        _chunk("a", "auth.py", "def login(user): authenticate(user) redirect home", "login"),
        _chunk("b", "billing.py", "def charge(card): stripe charge payment", "charge"),
        _chunk("c", "util.py", "def slug(text): return text.lower()", "slug"),
    ]
    r = BM25Retriever(chunks)
    results = r.search("fix login redirect", top_k=3)
    assert results
    assert results[0].chunk["chunk_id"] == "a"


def test_bm25_matches_camel_case_identifiers():
    chunks = [
        _chunk("a", "hooks.py", "def handleStripeWebhook(event): verify(event)", "handleStripeWebhook"),
        _chunk("b", "auth.py", "def login(user): pass", "login"),
    ]
    r = BM25Retriever(chunks)
    results = r.search("stripe webhook", top_k=2)
    assert results and results[0].chunk["chunk_id"] == "a"


def test_bm25_empty_query_and_corpus():
    assert BM25Retriever([]).search("anything") == []
    chunks = [_chunk("a", "x.py", "def f(): pass")]
    assert BM25Retriever(chunks).search("") == []


# --- Hybrid RRF -------------------------------------------------------------

class _FakeRetriever:
    def __init__(self, ranked: list[dict]):
        self._ranked = ranked

    def search(self, query: str, top_k: int = 8):
        return [RetrievalResult(c, 1.0 / (i + 1)) for i, c in enumerate(self._ranked[:top_k])]


def test_hybrid_rrf_fuses_rankings():
    a, b, c = _chunk("a", "a.py", "aaa"), _chunk("b", "b.py", "bbb"), _chunk("c", "c.py", "ccc")
    lexical = _FakeRetriever([a, b])
    semantic = _FakeRetriever([b, c])
    fused = HybridRetriever(lexical, semantic).search("q", top_k=3)
    # b appears in both rankings -> should win
    assert fused[0].chunk["chunk_id"] == "b"
    assert {r.chunk["chunk_id"] for r in fused} == {"a", "b", "c"}


def test_hybrid_survives_semantic_failure():
    a = _chunk("a", "a.py", "aaa")

    class Broken:
        def search(self, query, top_k=8):
            raise RuntimeError("model exploded")

    fused = HybridRetriever(_FakeRetriever([a]), Broken()).search("q", top_k=2)
    assert fused and fused[0].chunk["chunk_id"] == "a"


# --- Budget enforcement -----------------------------------------------------

def test_budget_is_hard_ceiling(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    big_body = "\n".join(f"    step_{i} = compute_login_redirect({i})" for i in range(400))
    (project / "auth.py").write_text(
        "def login_redirect(user):\n" + big_body + "\n    return user\n",
        encoding="utf-8",
    )
    config = XkitConfig()
    build_index(project, config)

    budget = 50
    result = retrieve_context(project, "fix login redirect", config, top_k=4, budget_tokens=budget)
    assert result["chunks"], "expected at least one (possibly truncated) chunk"
    assert result["event"]["retrieved_token_estimate"] <= budget * 1.2  # small estimator slack
    assert any("truncated" in c["text"] for c in result["chunks"]) or (
        result["event"]["retrieved_token_estimate"] <= budget
    )


def test_budget_skips_oversized_then_fits_smaller(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "big.py").write_text(
        "def login_redirect_big():\n" + "\n".join(f"    login_redirect_{i} = {i}" for i in range(300)),
        encoding="utf-8",
    )
    (project / "small.py").write_text(
        "def login_redirect_small():\n    return redirect('/login')\n",
        encoding="utf-8",
    )
    config = XkitConfig()
    build_index(project, config)

    result = retrieve_context(project, "login redirect", config, top_k=4, budget_tokens=120)
    assert result["event"]["retrieved_token_estimate"] <= 150
    files = {c["file"] for c in result["chunks"]}
    assert "small.py" in files
