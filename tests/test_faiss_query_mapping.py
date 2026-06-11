"""Regression test: FAISS query must resolve numeric ids via idmap, not position.

Before the fix, IndexIDMap assigned numeric ids starting at 1, but query() did
a positional ``ids[idx]`` lookup — returning the *wrong chunk* for every hit.
"""
from __future__ import annotations

import pytest

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from xkit.embeddings import FAISSEmbeddingStore  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_query_returns_correct_ids(tmp_path):
    store = FAISSEmbeddingStore(str(tmp_path / "faiss"))
    ns = "proj"
    # Three orthogonal vectors with distinct ids
    vecs = np.eye(3, dtype=np.float32)
    ids = ["chunk-alpha", "chunk-beta", "chunk-gamma"]
    metas = [{"file": f"f{i}.py"} for i in range(3)]
    store.upsert(ns, ids, vecs, metas, ["a", "b", "c"])

    # Query exactly along axis 1 -> must return chunk-beta first
    rows = store.query(ns, _unit([0.0, 1.0, 0.0]).reshape(1, -1), top_k=3)
    assert rows and rows[0]
    assert rows[0][0]["id"] == "chunk-beta"

    # And axis 0 -> chunk-alpha (the old positional bug returned chunk-beta here)
    rows = store.query(ns, _unit([1.0, 0.0, 0.0]).reshape(1, -1), top_k=1)
    assert rows[0][0]["id"] == "chunk-alpha"


def test_query_correct_after_removal(tmp_path):
    store = FAISSEmbeddingStore(str(tmp_path / "faiss"))
    ns = "proj"
    vecs = np.eye(4, dtype=np.float32)
    ids = ["c0", "c1", "c2", "c3"]
    store.upsert(ns, ids, vecs, [{} for _ in ids], ["", "", "", ""])
    store.remove(ns, ["c1"])  # creates a gap in the numeric id space

    rows = store.query(ns, _unit([0.0, 0.0, 1.0, 0.0]).reshape(1, -1), top_k=1)
    assert rows and rows[0], "expected a hit after removal"
    assert rows[0][0]["id"] == "c2"


def test_has_namespace(tmp_path):
    store = FAISSEmbeddingStore(str(tmp_path / "faiss"))
    assert not store.has_namespace("nope")
    store.upsert("yes", ["a"], np.eye(1, 8, dtype=np.float32), [{}], [""])
    assert store.has_namespace("yes")
