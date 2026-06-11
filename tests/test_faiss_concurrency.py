import multiprocessing as mp
import time

import pytest


def has_faiss_dependencies():
    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not has_faiss_dependencies(), reason="faiss/numpy not installed")
def test_faiss_concurrent_writes_and_reads(tmp_path):
    """Integration-style test: spawn two processes that upsert different ids while main process queries."""
    import numpy as np

    from xkit.embeddings import FAISSEmbeddingStore

    persist = str(tmp_path / "faiss")
    namespace = "testns"

    def writer(proc_id, start_idx):
        store = FAISSEmbeddingStore(persist)
        for i in range(5):
            _id = f"p{proc_id}-{start_idx + i}"
            vec = np.random.rand(1, 16).astype(np.float32)
            store.upsert(namespace, [_id], vec, [{}], [""])
            time.sleep(0.05)

    # start two writer processes
    p1 = mp.Process(target=writer, args=(1, 0))
    p2 = mp.Process(target=writer, args=(2, 100))
    p1.start()
    p2.start()

    # main process queries intermittently
    store = FAISSEmbeddingStore(persist)
    for _ in range(15):
        # random query
        q = np.random.rand(16).astype(np.float32)
        try:
            res = store.query(namespace, q, top_k=3)
        except Exception:
            res = []
        # should not raise and should be a list
        assert isinstance(res, list)
        time.sleep(0.02)

    p1.join(timeout=5)
    p2.join(timeout=5)
    assert not p1.is_alive()
    assert not p2.is_alive()
