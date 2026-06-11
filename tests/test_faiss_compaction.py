import pytest


def has_faiss():
    try:
        return True
    except Exception:
        return False


@pytest.mark.skipif(not has_faiss(), reason="faiss not installed")
def test_compact_and_remove(tmp_path):
    import numpy as np

    from xkit.embeddings import FAISSEmbeddingStore

    persist = str(tmp_path / "faiss")
    ns = "test"
    store = FAISSEmbeddingStore(persist)

    # upsert 6 vectors
    ids = [f"id{i}" for i in range(6)]
    vecs = np.random.rand(6, 8).astype(np.float32)
    metas = [{} for _ in ids]
    store.upsert(ns, ids, vecs, metas, [""] * len(ids))

    # remove a couple
    store.remove(ns, ["id1", "id4"])

    # compact
    store.compact(ns)

    # health check
    h = store.health(ns)
    assert h.get("exists") is True
    assert h.get("index_ok") is True
    # remaining ids should be 4
    assert h.get("ids_count") == 4

    # query should not error
    q = np.random.rand(8).astype(np.float32)
    res = store.query(ns, q, top_k=2)
    assert isinstance(res, list)
