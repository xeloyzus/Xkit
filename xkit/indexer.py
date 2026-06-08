from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .chunker import chunk_file
from .config import XkitConfig
from .files import iter_project_files, read_text_safe, sha256_text
from .store import ensure_store, read_json, read_jsonl, write_json, write_jsonl
from .token_estimator import estimate_tokens
from .logging import get_logger

_logger = get_logger(__name__)


def _progress(msg: str, show: bool = False):
    """Simple progress output. In a real app, replace with rich.progress or tqdm."""
    _logger.debug(msg)
    if show:
        print(f"  {msg}", file=sys.stderr)


def build_index(project_root: Path, config: XkitConfig, show_progress: bool = False) -> dict:
    started = time.time()
    store = ensure_store(project_root, config)
    chunks: list[dict] = []
    files_meta: dict = {}
    full_repo_tokens = 0

    # Count files first for progress
    all_files = list(iter_project_files(project_root, config))
    total = len(all_files)
    _progress(f"Indexing {total} files...", show_progress)

    # Try to initialize a persistent embedding store (optional)
    try:
        from .embeddings import FAISSEmbeddingStore
        _embed_store = FAISSEmbeddingStore(str(project_root / config.index_dir_name / "faiss"))
    except Exception:
        _embed_store = None

    def _process(path: Path):
        text = read_text_safe(path)
        if text is None:
            raise RuntimeError("unable to read file")
        rel = str(path.relative_to(project_root))
        file_hash = sha256_text(text)
        file_tokens = estimate_tokens(text)
        file_chunks = chunk_file(project_root, path, text, config.max_chunk_chars, config.min_chunk_chars, config.overlap_lines)
        return rel, file_hash, file_tokens, file_chunks

    # Parallelize file processing to speed up indexing on large repos.
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_process, path): path for path in all_files}
        processed = 0
        for fut in as_completed(futures):
            processed += 1
            path = futures[fut]
            try:
                rel, file_hash, file_tokens, file_chunks = fut.result()
            except Exception:
                # Skip files that raised during processing
                continue
            full_repo_tokens += file_tokens
            chunks.extend([c.to_dict() for c in file_chunks])
            files_meta[rel] = {
                "hash": file_hash,
                "token_estimate": file_tokens,
                "chunk_count": len(file_chunks),
                "size_chars": sum(len(c.text) for c in file_chunks),
            }
            if show_progress and (processed % 50 == 0 or processed == total):
                _progress(f"  [{processed}/{total}] files processed, {len(chunks)} chunks so far", show_progress)

    index = {
        "version": 1,
        "project_root": str(project_root),
        "files": files_meta,
        "chunk_count": len(chunks),
        "file_count": len(files_meta),
        "full_repo_token_estimate": full_repo_tokens,
        "updated_at": int(time.time()),
    }
    write_json(store / "index.json", index)
    write_jsonl(store / "chunks.jsonl", chunks)

    # Try to compute embeddings and persist to FAISS (optional)
    if _embed_store is not None and chunks:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
            texts = [c.get("text", "") + "\nFile: " + c.get("file", "") + ("\nSymbol: " + str(c.get("symbol", "")) if c.get("symbol") else "") for c in chunks]
            ids = [c.get("chunk_id") for c in chunks]
            metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
            embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embs = np.asarray(embs, dtype=np.float32)
            t0 = time.time()
            _embed_store.upsert(str(project_root.name), ids, embs, metadatas, texts)
            upsert_dur = time.time() - t0
            try:
                metrics = read_json(store / "metrics.json", {})
                metrics.setdefault("embedding_runs", [])
                metrics["embedding_runs"].append({
                    "type": "full",
                    "duration_sec": round(upsert_dur, 3),
                    "chunk_count": len(ids),
                    "updated_at": int(time.time()),
                })
                write_json(store / "metrics.json", metrics)
            except Exception:
                pass
        except Exception as e:
            _logger.warning("Embedding upsert failed: %s", e)

    metrics = read_json(store / "metrics.json", {})
    metrics.setdefault("index_runs", [])
    metrics["index_runs"].append({
        "type": "full",
        "duration_sec": round(time.time() - started, 3),
        "file_count": len(files_meta),
        "chunk_count": len(chunks),
        "full_repo_token_estimate": full_repo_tokens,
        "updated_at": int(time.time()),
    })
    write_json(store / "metrics.json", metrics)

    elapsed = time.time() - started
    _progress(f"Done in {elapsed:.2f}s — {len(files_meta)} files, {len(chunks)} chunks", show_progress)
    return index


def update_changed_files(project_root: Path, config: XkitConfig, show_progress: bool = False) -> dict:
    started = time.time()
    store = ensure_store(project_root, config)
    index = read_json(store / "index.json", None)
    if not index:
        _progress("No existing index found — performing full index instead", show_progress)
        return build_index(project_root, config, show_progress)

    existing_chunks = read_jsonl(store / "chunks.jsonl")
    files_meta = index.get("files", {})
    old_by_file = {rel: meta for rel, meta in files_meta.items()}

    changed_files: list[str] = []
    current_files: dict = {}
    new_chunks_by_file: dict = {}
    full_repo_tokens = 0

    all_files = list(iter_project_files(project_root, config))
    total = len(all_files)
    _progress(f"Checking {total} files for changes...", show_progress)

    # Try to initialize a persistent embedding store (optional)
    try:
        from .embeddings import FAISSEmbeddingStore
        _embed_store = FAISSEmbeddingStore(str(project_root / config.index_dir_name / "faiss"))
    except Exception:
        _embed_store = None

    def _process(path: Path):
        text = read_text_safe(path)
        if text is None:
            return None
        rel = str(path.relative_to(project_root))
        file_hash = sha256_text(text)
        file_tokens = estimate_tokens(text)
        file_chunks = chunk_file(project_root, path, text, config.max_chunk_chars, config.min_chunk_chars, config.overlap_lines)
        return rel, file_hash, file_tokens, file_chunks

    # Parallelize file checking and chunking
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_process, path): path for path in all_files}

        processed = 0
        for fut in as_completed(futures):
            processed += 1
            path = futures[fut]
            try:
                result = fut.result()
                if result is None:
                    continue
                rel, file_hash, file_tokens, file_chunks = result
            except Exception:
                continue
            full_repo_tokens += file_tokens
            old = old_by_file.get(rel)
            current_files[rel] = {
                "hash": file_hash,
                "token_estimate": file_tokens,
                "chunk_count": old.get("chunk_count", 0) if old else 0,
                "size_chars": sum(len(c.text) for c in file_chunks) if file_chunks else 0,
            }
            if not old or old.get("hash") != file_hash:
                changed_files.append(rel)
                new_chunks_by_file[rel] = [c.to_dict() for c in file_chunks]
                current_files[rel]["chunk_count"] = len(file_chunks)

            if show_progress and (processed % 50 == 0 or processed == total):
                _progress(f"  [{processed}/{total}] checked, {len(changed_files)} changed", show_progress)

    deleted_files = sorted(set(old_by_file) - set(current_files))
    changed_set = set(changed_files) | set(deleted_files)
    kept_chunks = [c for c in existing_chunks if c["file"] not in changed_set]
    new_chunks = []
    for rows in new_chunks_by_file.values():
        new_chunks.extend(rows)

    all_chunks = kept_chunks + new_chunks
    index.update({
        "files": current_files,
        "chunk_count": len(all_chunks),
        "file_count": len(current_files),
        "full_repo_token_estimate": full_repo_tokens,
        "updated_at": int(time.time()),
    })
    write_json(store / "index.json", index)
    write_jsonl(store / "chunks.jsonl", all_chunks)

    metrics = read_json(store / "metrics.json", {})
    metrics.setdefault("index_runs", [])
    metrics["index_runs"].append({
        "type": "incremental",
        "duration_sec": round(time.time() - started, 3),
        "changed_files": changed_files,
        "deleted_files": deleted_files,
        "changed_file_count": len(changed_files),
        "deleted_file_count": len(deleted_files),
        "reembedded_chunk_count": len(new_chunks),
        "total_chunk_count": len(all_chunks),
        "full_repo_token_estimate": full_repo_tokens,
        "updated_at": int(time.time()),
    })
    write_json(store / "metrics.json", metrics)

    # Try to compute embeddings for new chunks and persist (optional)
    if _embed_store is not None and new_chunks:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
            texts = [c.get("text", "") + "\nFile: " + c.get("file", "") + ("\nSymbol: " + str(c.get("symbol", "")) if c.get("symbol") else "") for c in new_chunks]
            ids = [c.get("chunk_id") for c in new_chunks]
            metadatas = [{k: v for k, v in c.items() if k != "text"} for c in new_chunks]
            embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embs = np.asarray(embs, dtype=np.float32)
            t0 = time.time()
            _embed_store.upsert(str(project_root.name), ids, embs, metadatas, texts)
            upsert_dur = time.time() - t0
            try:
                metrics = read_json(store / "metrics.json", {})
                metrics.setdefault("embedding_runs", [])
                metrics["embedding_runs"].append({
                    "type": "incremental",
                    "duration_sec": round(upsert_dur, 3),
                    "chunk_count": len(ids),
                    "updated_at": int(time.time()),
                })
                write_json(store / "metrics.json", metrics)
            except Exception:
                pass
        except Exception as e:
            _logger.warning("Embedding upsert failed (incremental): %s", e)

    elapsed = time.time() - started
    _progress(
        f"Done in {elapsed:.2f}s — {len(changed_files)} changed, {len(deleted_files)} deleted, "
        f"{len(all_chunks)} total chunks",
        show_progress,
    )
    return index
