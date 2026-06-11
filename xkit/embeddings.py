from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

# Re-entrant lock machinery shared by all FAISSEmbeddingStore instances:
# one RLock per lock-file path (intra-process serialization) and a
# thread-local depth counter so only the outermost frame takes the OS lock.
_LOCK_REGISTRY: dict[str, threading.RLock] = {}
_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_DEPTHS = threading.local()


class PersistentEmbeddingStore:
    """Minimal adapter: persistent embedding store using Chroma (optional)."""

    def __init__(self, persist_dir: str | None = None):
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as e:
            raise ImportError("chromadb is required for PersistentEmbeddingStore") from e
        persist = persist_dir or ".xkit/chroma"
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist))
        self._client = client

    def upsert(self, namespace: str, ids: List[str], embeddings, metadatas: List[dict], documents: List[str]):
        col = self._client.get_or_create_collection(namespace)
        col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, namespace: str, query_embeddings, top_k: int = 8):
        col = self._client.get_collection(namespace)
        return col.query(query_embeddings=query_embeddings, n_results=top_k)


class FAISSEmbeddingStore:
    """FAISS-backed embedding store with id mapping, compaction, backups and cross-platform locks.

    Files per namespace:
      - index.faiss
      - ids.npy        (ordered list of original ids)
      - vectors.npy    (stored vectors in same order)
      - meta.npy       (optional metadata list)
      - idmap.json     (original id -> numeric id)
      - next_id.json   (next numeric id to assign)
      - free_ids.npy   (numeric ids available for reuse)
      - lock           (file used for inter-process locking)
    """

    def __init__(self, persist_dir: str | None = None):
        try:
            import faiss
            import numpy as np
        except Exception as e:
            raise ImportError("faiss and numpy are required for FAISSEmbeddingStore") from e
        self._faiss = faiss
        self._np = np
        self._root = Path(persist_dir or ".xkit/faiss")
        self._root.mkdir(parents=True, exist_ok=True)

    def _ensure_parent(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @contextmanager
    def _file_lock(self, lock_path: str, mode: str = "ex"):
        """Cross-platform, *re-entrant* file lock.

        Intra-process: a per-path threading.RLock serializes threads and allows
        nested acquisition (remove() -> _load() -> rebuild() all lock the same
        path). Inter-process: the OS lock (flock/msvcrt) is taken only by the
        outermost frame — taking flock twice on separate fds of the same file
        deadlocks the process, which is exactly the bug this fixes.
        """
        self._ensure_parent(Path(lock_path))
        with _LOCK_REGISTRY_GUARD:
            rlock = _LOCK_REGISTRY.setdefault(lock_path, threading.RLock())
        rlock.acquire()
        depths = getattr(_LOCK_DEPTHS, "by_path", None)
        if depths is None:
            depths = _LOCK_DEPTHS.by_path = {}
        outermost = depths.get(lock_path, 0) == 0
        depths[lock_path] = depths.get(lock_path, 0) + 1
        f = None
        try:
            if outermost:
                f = open(lock_path, "a+b")
                if os.name == "nt":
                    try:
                        import msvcrt
                        if mode == "sh":
                            msvcrt.locking(f.fileno(), msvcrt.LK_RLCK, 1)
                        else:
                            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    except Exception:
                        try:
                            import portalocker
                            portalocker.lock(f, portalocker.LOCK_EX if mode != "sh" else portalocker.LOCK_SH)
                        except Exception:
                            pass  # last resort: intra-process lock only
                else:
                    import fcntl
                    if mode == "sh":
                        fcntl.flock(f, fcntl.LOCK_SH)
                    else:
                        fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            depths[lock_path] -= 1
            if outermost and f is not None:
                try:
                    if os.name == "nt":
                        try:
                            import msvcrt
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                        except Exception:
                            try:
                                import portalocker
                                portalocker.unlock(f)
                            except Exception:
                                pass
                    else:
                        try:
                            import fcntl
                            fcntl.flock(f, fcntl.LOCK_UN)
                        except Exception:
                            pass
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass
            rlock.release()

    def _ns_paths(self, namespace: str) -> Tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
        ns = self._root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        return (
            ns / "index.faiss",
            ns / "ids.npy",
            ns / "meta.npy",
            ns / "vectors.npy",
            ns / "lock",
            ns / "idmap.json",
            ns / "next_id.json",
            ns / "free_ids.npy",
        )

    def _backup_files(self, ns: Path):
        try:
            bak = ns / "backups" / str(int(time.time()))
            bak.mkdir(parents=True, exist_ok=True)
            for p in ns.iterdir():
                if p.is_file() and p.name != "backups":
                    try:
                        shutil.copy2(p, bak / p.name)
                    except Exception:
                        pass
        except Exception:
            pass

    def _load(self, namespace: str):
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, next_id_path, free_ids_path = self._ns_paths(namespace)
        if not idx_path.exists() or not ids_path.exists() or not vecs_path.exists():
            return None, None, None, None, None, None, None
        with self._file_lock(str(lock_path), mode="sh"):
            try:
                index = self._faiss.read_index(str(idx_path))
            except Exception:
                index = None
            try:
                ids = self._np.load(str(ids_path), allow_pickle=True).tolist()
            except Exception:
                ids = []
            meta = None
            if meta_path.exists():
                try:
                    meta = self._np.load(str(meta_path), allow_pickle=True).tolist()
                except Exception:
                    meta = None
            vectors = None
            if vecs_path.exists():
                try:
                    vectors = self._np.load(str(vecs_path), allow_pickle=True)
                except Exception:
                    vectors = None
            idmap = None
            next_id = None
            free_ids = None
            try:
                if idmap_path.exists():
                    idmap = json.loads(idmap_path.read_text())
            except Exception:
                idmap = None
            try:
                if next_id_path.exists():
                    next_id = json.loads(next_id_path.read_text())
            except Exception:
                next_id = None
            try:
                if free_ids_path.exists():
                    free_ids = self._np.load(str(free_ids_path), allow_pickle=True).tolist()
            except Exception:
                free_ids = None
        return index, ids, meta, vectors, idmap, next_id, free_ids

    def _save(self, namespace: str, index, ids, vectors=None, meta=None, idmap: Optional[dict] = None, next_id: Optional[int] = None, free_ids: Optional[List[int]] = None):
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, next_id_path, free_ids_path = self._ns_paths(namespace)
        ns = idx_path.parent
        # backup prior state
        try:
            self._backup_files(ns)
        except Exception:
            pass
        tmp_idx = str(idx_path) + ".tmp"
        tmp_ids = str(ids_path) + ".tmp"
        tmp_vecs = str(vecs_path) + ".tmp"
        tmp_meta = str(meta_path) + ".tmp"
        tmp_idmap = str(idmap_path) + ".tmp"
        tmp_next = str(next_id_path) + ".tmp"
        tmp_free = str(free_ids_path) + ".tmp"
        try:
            self._faiss.write_index(index, tmp_idx)
            os.replace(tmp_idx, str(idx_path))
        except Exception:
            try:
                self._faiss.write_index(index, str(idx_path))
            except Exception:
                pass
        try:
            self._np.save(tmp_ids, self._np.array(ids, dtype=object))
            os.replace(tmp_ids, str(ids_path))
        except Exception:
            try:
                self._np.save(str(ids_path), self._np.array(ids, dtype=object))
            except Exception:
                pass
        if vectors is not None:
            try:
                self._np.save(tmp_vecs, vectors)
                os.replace(tmp_vecs, str(vecs_path))
            except Exception:
                try:
                    self._np.save(str(vecs_path), vectors)
                except Exception:
                    pass
        if meta is not None:
            try:
                self._np.save(tmp_meta, self._np.array(meta, dtype=object))
                os.replace(tmp_meta, str(meta_path))
            except Exception:
                try:
                    self._np.save(str(meta_path), self._np.array(meta, dtype=object))
                except Exception:
                    pass
        if idmap is not None:
            try:
                Path(tmp_idmap).write_text(json.dumps(idmap))
                os.replace(tmp_idmap, str(idmap_path))
            except Exception:
                try:
                    idmap_path.write_text(json.dumps(idmap))
                except Exception:
                    pass
        if next_id is not None:
            try:
                Path(tmp_next).write_text(json.dumps(next_id))
                os.replace(tmp_next, str(next_id_path))
            except Exception:
                try:
                    next_id_path.write_text(json.dumps(next_id))
                except Exception:
                    pass
        if free_ids is not None:
            try:
                self._np.save(tmp_free, self._np.array(free_ids, dtype=object))
                os.replace(tmp_free, str(free_ids_path))
            except Exception:
                try:
                    self._np.save(str(free_ids_path), self._np.array(free_ids, dtype=object))
                except Exception:
                    pass

    def upsert(self, namespace: str, ids: List[str], embeddings, metadatas: List[dict], documents: List[str]):
        arr = self._np.asarray(embeddings, dtype=self._np.float32)
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, next_id_path, free_ids_path = self._ns_paths(namespace)

        with self._file_lock(str(lock_path), mode="ex"):
            loaded = self._load(namespace)
            if loaded[0] is None:
                # create fresh index
                dim = arr.shape[1] if arr.size else 0
                if dim == 0:
                    return
                index = self._faiss.IndexFlatIP(dim)
                id_map = self._faiss.IndexIDMap(index)
                norms = self._np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arrn = arr / norms
                numeric_ids = self._np.arange(1, len(ids) + 1, dtype=self._np.int64)
                id_map.add_with_ids(arrn, numeric_ids)
                idmap = {ids[i]: int(numeric_ids[i]) for i in range(len(ids))}
                next_id = int(len(ids) + 1)
                free_ids = []
                self._save(namespace, id_map, ids, vectors=arrn, meta=metadatas, idmap=idmap, next_id=next_id, free_ids=free_ids)
                return

            index, old_ids, old_meta, old_vectors, idmap, next_id, free_ids = loaded
            existing_set = set(old_ids)
            replace_ids = [i for i in ids if i in existing_set]

            if replace_ids:
                # conservative full rebuild
                id_to_vec = {}
                id_to_meta = {}
                for i, _id in enumerate(old_ids):
                    id_to_vec[_id] = old_vectors[i]
                    id_to_meta[_id] = (old_meta[i] if old_meta else {})
                for i, _id in enumerate(ids):
                    id_to_vec[_id] = arr[i]
                    id_to_meta[_id] = metadatas[i] if metadatas else {}
                final_ids = list(id_to_vec.keys())
                final_vectors = self._np.stack([id_to_vec[_id] for _id in final_ids]).astype(self._np.float32)
                dim = final_vectors.shape[1] if final_vectors.size else 0
                if dim == 0:
                    return
                base = self._faiss.IndexFlatIP(dim)
                id_map = self._faiss.IndexIDMap(base)
                norms = self._np.linalg.norm(final_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                final_vectors = final_vectors / norms
                numeric_ids = self._np.arange(1, len(final_ids) + 1, dtype=self._np.int64)
                id_map.add_with_ids(final_vectors, numeric_ids)
                new_idmap = {final_ids[i]: int(numeric_ids[i]) for i in range(len(final_ids))}
                next_id = int(len(final_ids) + 1)
                free_ids = []
                self._save(namespace, id_map, final_ids, vectors=final_vectors, meta=[id_to_meta.get(i) for i in final_ids], idmap=new_idmap, next_id=next_id, free_ids=free_ids)
                return

            # append only new ids
            try:
                if not isinstance(index, self._faiss.IndexIDMap):
                    base = index
                    index = self._faiss.IndexIDMap(base)
            except Exception:
                base = index
                index = self._faiss.IndexIDMap(base)

            if idmap is None:
                idmap = {old_ids[i]: i + 1 for i in range(len(old_ids))}
            if next_id is None:
                next_id = max(idmap.values()) + 1 if idmap else len(old_ids) + 1
            if free_ids is None:
                free_ids = []

            new_idx_positions = []
            new_vectors = []
            for i, _id in enumerate(ids):
                if _id in existing_set:
                    continue
                new_vectors.append(arr[i])
                if free_ids:
                    nid = int(free_ids.pop())
                else:
                    nid = int(next_id)
                    next_id += 1
                new_idx_positions.append(nid)
                old_ids.append(_id)
                old_meta.append(metadatas[i] if metadatas else {})
                idmap[_id] = int(nid)

            if new_vectors:
                new_vectors = self._np.stack(new_vectors).astype(self._np.float32)
                norms = self._np.linalg.norm(new_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_vectors = new_vectors / norms
                numeric_ids = self._np.array(new_idx_positions, dtype=self._np.int64)
                index.add_with_ids(new_vectors, numeric_ids)
                if old_vectors is None:
                    all_vectors = new_vectors
                else:
                    all_vectors = self._np.vstack([old_vectors, new_vectors])
                try:
                    self._save(namespace, index, old_ids, vectors=all_vectors, meta=old_meta, idmap=idmap, next_id=next_id, free_ids=free_ids)
                except Exception:
                    self._save(namespace, index, old_ids, vectors=all_vectors, meta=old_meta)
            return

    def has_namespace(self, namespace: str) -> bool:
        """True if a queryable FAISS index exists on disk for this namespace."""
        idx_path, *_ = self._ns_paths(namespace)
        return idx_path.exists()

    def query(self, namespace: str, query_embeddings, top_k: int = 8):
        q = self._np.asarray(query_embeddings, dtype=self._np.float32)
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, *_ = self._ns_paths(namespace)
        if not idx_path.exists():
            return []
        with self._file_lock(str(lock_path), mode="sh"):
            try:
                index = self._faiss.read_index(str(idx_path))
            except Exception:
                return []
            try:
                ids = self._np.load(str(ids_path), allow_pickle=True).tolist()
            except Exception:
                ids = []
            idmap = None
            try:
                if idmap_path.exists():
                    idmap = json.loads(idmap_path.read_text(encoding="utf-8"))
            except Exception:
                idmap = None
        # The index stores *numeric* ids (assigned starting at 1, with gaps after
        # removals), so search results must be resolved through the reverse idmap —
        # never by list position.
        if idmap:
            reverse = {int(v): k for k, v in idmap.items()}
        else:
            # Legacy stores without an idmap used sequential ids starting at 1.
            reverse = {i + 1: _id for i, _id in enumerate(ids)}
        if q.ndim == 1:
            q = q.reshape(1, -1)
        norms = self._np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q = q / norms
        scores_arr, ids_arr = index.search(q, top_k)
        results = []
        for row_scores, row_idx in zip(scores_arr, ids_arr, strict=True):
            row = []
            for score, numeric_id in zip(row_scores, row_idx, strict=True):
                original_id = reverse.get(int(numeric_id))
                if numeric_id < 0 or original_id is None:
                    continue
                row.append({"id": original_id, "score": float(score)})
            results.append(row)
        return results

    def rebuild(self, namespace: str, ids: List[str], embeddings, metadatas: List[dict]):
        arr = self._np.asarray(embeddings, dtype=self._np.float32)
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, next_id_path, free_ids_path = self._ns_paths(namespace)
        with self._file_lock(str(lock_path), mode="ex"):
            dim = arr.shape[1] if arr.size else 0
            if dim == 0:
                return
            base = self._faiss.IndexFlatIP(dim)
            id_map = self._faiss.IndexIDMap(base)
            norms = self._np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arrn = arr / norms
            numeric_ids = self._np.arange(1, len(ids) + 1, dtype=self._np.int64)
            id_map.add_with_ids(arrn, numeric_ids)
            idmap = {ids[i]: int(numeric_ids[i]) for i in range(len(ids))}
            next_id = int(len(ids) + 1)
            free_ids = []
            self._save(namespace, id_map, ids, vectors=arrn, meta=metadatas, idmap=idmap, next_id=next_id, free_ids=free_ids)

    def remove(self, namespace: str, remove_ids: List[str]):
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, next_id_path, free_ids_path = self._ns_paths(namespace)
        with self._file_lock(str(lock_path), mode="ex"):
            loaded = self._load(namespace)
            if loaded[0] is None:
                return
            index, ids, meta, vectors, idmap, next_id, free_ids = loaded
            keep = [i for i in ids if i not in set(remove_ids)]
            if not keep:
                for p in [idx_path, ids_path, vecs_path, meta_path, idmap_path, next_id_path, free_ids_path]:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
                return
            keep_idx = [ids.index(k) for k in keep]
            new_ids = [ids[i] for i in keep_idx]
            new_meta = [meta[i] if meta else {} for i in keep_idx]
            new_vectors = vectors[keep_idx]
            # record freed numeric ids
            if idmap is None:
                idmap = {ids[i]: i + 1 for i in range(len(ids))}
            if free_ids is None:
                free_ids = []
            for rid in remove_ids:
                if rid in idmap:
                    free_ids.append(idmap[rid])
                    try:
                        del idmap[rid]
                    except Exception:
                        pass
            try:
                self._np.save(str(free_ids_path), self._np.array(free_ids, dtype=object))
                idmap_path.write_text(json.dumps(idmap))
            except Exception:
                pass
            self.rebuild(namespace, new_ids, new_vectors, new_meta)

    def compact(self, namespace: str):
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, next_id_path, free_ids_path = self._ns_paths(namespace)
        with self._file_lock(str(lock_path), mode="ex"):
            loaded = self._load(namespace)
            if loaded[0] is None:
                return
            index, ids, meta, vectors, idmap, next_id, free_ids = loaded
            if vectors is None:
                return
            final_ids = list(ids)
            dim = vectors.shape[1]
            base = self._faiss.IndexFlatIP(dim)
            id_map = self._faiss.IndexIDMap(base)
            norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs_n = vectors / norms
            numeric_ids = self._np.arange(1, len(final_ids) + 1, dtype=self._np.int64)
            id_map.add_with_ids(vecs_n, numeric_ids)
            new_idmap = {final_ids[i]: int(numeric_ids[i]) for i in range(len(final_ids))}
            next_id = int(len(final_ids) + 1)
            free_ids = []
            self._save(namespace, id_map, final_ids, vectors=vecs_n, meta=meta, idmap=new_idmap, next_id=next_id, free_ids=free_ids)

    def health(self, namespace: str) -> dict:
        """Return a small health report for the namespace."""
        idx_path, ids_path, meta_path, vecs_path, lock_path, idmap_path, next_id_path, free_ids_path = self._ns_paths(namespace)
        report = {"exists": False}
        if not idx_path.exists():
            return report
        report["exists"] = True
        with self._file_lock(str(lock_path), mode="sh"):
            try:
                self._faiss.read_index(str(idx_path))
                report["index_ok"] = True
            except Exception as e:
                report["index_ok"] = False
                report["index_error"] = str(e)
            try:
                ids = self._np.load(str(ids_path), allow_pickle=True).tolist()
                report["ids_count"] = len(ids)
            except Exception as e:
                report["ids_count"] = 0
                report["ids_error"] = str(e)
            try:
                vecs = self._np.load(str(vecs_path), allow_pickle=True)
                report["vectors_shape"] = list(vecs.shape)
            except Exception as e:
                report["vectors_shape"] = None
                report["vectors_error"] = str(e)
            try:
                if idmap_path.exists():
                    report["idmap_len"] = len(json.loads(idmap_path.read_text()))
            except Exception:
                report["idmap_len"] = None
        return report


__all__ = ["PersistentEmbeddingStore", "FAISSEmbeddingStore"]
