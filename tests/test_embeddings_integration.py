import tempfile

import pytest


def test_faiss_store_upsert_query():
    pytest.importorskip("faiss")
    np = pytest.importorskip("numpy")
    from xkit.embeddings import FAISSEmbeddingStore

    td = tempfile.mkdtemp()
    store = FAISSEmbeddingStore(td)
    ids = ["a", "b"]
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    metas = [{"id": "a"}, {"id": "b"}]
    docs = ["doc a", "doc b"]
    store.upsert("testns", ids, vecs, metas, docs)
    res = store.query("testns", vecs[0], top_k=2)
    assert res and isinstance(res, list)


def test_tree_sitter_chunking():
    pytest.importorskip("tree_sitter")
    from pathlib import Path

    from xkit.chunker import chunk_file

    txt = """
def foo():
    return 1

def bar():
    return 2
"""
    p = Path(tempfile.mkdtemp()) / "a.py"
    p.write_text(txt, encoding="utf-8")
    chunks = chunk_file(p.parent, p, txt)
    assert isinstance(chunks, list)
