from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from .agent_context import retrieve_context, format_context_markdown
from .config import XkitConfig
from .indexer import build_index, update_changed_files
from .metrics import load_metrics, format_metrics_report
from .embeddings import FAISSEmbeddingStore


def positive_int(value: str) -> int:
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return i


def _load_config(project_root: Path) -> XkitConfig:
    """Load config from .xkit/config.toml, falling back to defaults."""
    return XkitConfig.load(project_root)


def cmd_init(args) -> int:
    """Initialize a project: create .xkit/ directory with default config."""
    root = Path(args.project).resolve()
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 2
    config = XkitConfig()
    path = config.save_default(root)
    print(json.dumps({
        "status": "initialized",
        "project": str(root),
        "config": str(path),
    }, indent=2))
    return 0


def cmd_index(args) -> int:
    root = Path(args.project).resolve()
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 2
    config = _load_config(root)
    index = build_index(root, config, show_progress=True)
    print(json.dumps({
        "status": "indexed",
        "project": str(root),
        "files": index["file_count"],
        "chunks": index["chunk_count"],
        "full_repo_token_estimate": index["full_repo_token_estimate"],
    }, indent=2))
    return 0


def cmd_update(args) -> int:
    root = Path(args.project).resolve()
    config = _load_config(root)
    index = update_changed_files(root, config, show_progress=True)
    print(json.dumps({
        "status": "updated",
        "project": str(root),
        "files": index["file_count"],
        "chunks": index["chunk_count"],
        "full_repo_token_estimate": index["full_repo_token_estimate"],
    }, indent=2))
    return 0


def cmd_retrieve(args) -> int:
    root = Path(args.project).resolve()
    config = _load_config(root)
    top_k = args.top_k or config.default_top_k
    budget = args.budget or config.default_budget_tokens
    try:
        result = retrieve_context(root, args.task, config, top_k, budget)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(format_context_markdown(result))
    return 0


def cmd_metrics(args) -> int:
    root = Path(args.project).resolve()
    config = _load_config(root)
    data = load_metrics(root, config)
    if args.format == "json":
        print(json.dumps(data, indent=2))
    else:
        print(format_metrics_report(data))
    return 0


def cmd_embeddings_rebuild(args) -> int:
    """Rebuild FAISS index from stored vectors for a namespace."""
    root = Path(args.project).resolve()
    store = None
    try:
        store = FAISSEmbeddingStore(str(root / ".xkit" / "faiss"))
    except Exception as e:
        print(f"FAISS not available: {e}", file=sys.stderr)
        return 2
    namespace = args.namespace
    # load existing vectors
    loaded = store._load(namespace)
    if not loaded or loaded[1] is None or loaded[3] is None:
        print(f"No stored vectors for namespace: {namespace}", file=sys.stderr)
        return 2
    _, ids, meta, vectors, *_ = loaded
    store.rebuild(namespace, ids, vectors, meta or [])
    print(json.dumps({"status": "rebuilt", "namespace": namespace}, indent=2))
    return 0


def cmd_embeddings_remove(args) -> int:
    """Remove ids from FAISS namespace."""
    root = Path(args.project).resolve()
    try:
        store = FAISSEmbeddingStore(str(root / ".xkit" / "faiss"))
    except Exception as e:
        print(f"FAISS not available: {e}", file=sys.stderr)
        return 2
    namespace = args.namespace
    ids = args.ids
    store.remove(namespace, ids)
    print(json.dumps({"status": "removed", "namespace": namespace, "ids": ids}, indent=2))
    return 0


def cmd_embeddings_compact(args) -> int:
    root = Path(args.project).resolve()
    try:
        store = FAISSEmbeddingStore(str(root / ".xkit" / "faiss"))
    except Exception as e:
        print(f"FAISS not available: {e}", file=sys.stderr)
        return 2
    namespace = args.namespace
    store.compact(namespace)
    print(json.dumps({"status": "compacted", "namespace": namespace}, indent=2))
    return 0


def cmd_embeddings_health(args) -> int:
    root = Path(args.project).resolve()
    try:
        store = FAISSEmbeddingStore(str(root / ".xkit" / "faiss"))
    except Exception as e:
        print(f"FAISS not available: {e}", file=sys.stderr)
        return 2
    namespace = args.namespace
    report = store.health(namespace)
    print(json.dumps(report, indent=2))
    return 0


def cmd_embeddings_export(args) -> int:
    root = Path(args.project).resolve()
    out = Path(args.dest).resolve()
    try:
        store = FAISSEmbeddingStore(str(root / ".xkit" / "faiss"))
    except Exception as e:
        print(f"FAISS not available: {e}", file=sys.stderr)
        return 2
    namespace = args.namespace
    loaded = store._load(namespace)
    if not loaded or loaded[1] is None or loaded[3] is None:
        print(f"No data to export for namespace: {namespace}", file=sys.stderr)
        return 2
    _, ids, meta, vectors, idmap, next_id, free_ids = loaded
    tmp = out.with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        # save arrays and json
        import numpy as np

        np.save(str(tmp / "ids.npy"), np.array(ids, dtype=object))
        np.save(str(tmp / "vectors.npy"), vectors)
        if meta is not None:
            np.save(str(tmp / "meta.npy"), np.array(meta, dtype=object))
        if idmap is not None:
            (tmp / "idmap.json").write_text(json.dumps(idmap))
        if next_id is not None:
            (tmp / "next_id.json").write_text(json.dumps(next_id))
        if free_ids is not None:
            np.save(str(tmp / "free_ids.npy"), np.array(free_ids, dtype=object))
        # create archive
        archive = str(out) if out.suffix in (".zip", ".tar", ".gz") else str(out) + ".tar.gz"
        shutil.make_archive(str(Path(archive).with_suffix("") ), 'gztar', root_dir=str(tmp))
        # cleanup tmp dir
        shutil.rmtree(str(tmp))
        print(json.dumps({"status": "exported", "path": archive}, indent=2))
        return 0
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        return 2


def cmd_embeddings_import(args) -> int:
    root = Path(args.project).resolve()
    src = Path(args.src).resolve()
    try:
        store = FAISSEmbeddingStore(str(root / ".xkit" / "faiss"))
    except Exception as e:
        print(f"FAISS not available: {e}", file=sys.stderr)
        return 2
    namespace = args.namespace
    import tarfile
    import numpy as np

    tmp = Path(".tmp_xkit_import")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(str(src), 'r:gz') as t:
            t.extractall(path=str(tmp))
        ids = np.load(str(tmp / "ids.npy"), allow_pickle=True).tolist()
        vectors = np.load(str(tmp / "vectors.npy"), allow_pickle=True)
        meta = None
        if (tmp / "meta.npy").exists():
            meta = np.load(str(tmp / "meta.npy"), allow_pickle=True).tolist()
        store.rebuild(namespace, ids, vectors, meta or [])
        shutil.rmtree(tmp)
        print(json.dumps({"status": "imported", "namespace": namespace}, indent=2))
        return 0
    except Exception as e:
        print(f"Import failed: {e}", file=sys.stderr)
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass
        return 2


def cmd_embeddings_cleanup(args) -> int:
    root = Path(args.project).resolve()
    days = args.retention_days
    base = root / ".xkit" / "faiss"
    cutoff = time.time() - (days * 24 * 3600)
    removed = 0
    for ns in base.iterdir():
        bak = ns / "backups"
        if not bak.exists():
            continue
        for tdir in bak.iterdir():
            try:
                ts = int(tdir.name)
            except Exception:
                continue
            if ts < cutoff:
                try:
                    shutil.rmtree(tdir)
                    removed += 1
                except Exception:
                    pass
    print(json.dumps({"status": "cleanup_complete", "removed": removed}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xkit",
        description="Xkit: Lower AI coding-agent token costs with project chunking, retrieval, and metrics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize .xkit/ config directory for a project")
    p_init.add_argument("project")
    p_init.set_defaults(func=cmd_init)

    p_index = sub.add_parser("index", help="Build the full project index once")
    p_index.add_argument("project")
    p_index.set_defaults(func=cmd_index)

    p_update = sub.add_parser("update", help="Incrementally update index for changed files")
    p_update.add_argument("project")
    p_update.set_defaults(func=cmd_update)

    p_retrieve = sub.add_parser("retrieve", help="Retrieve a small context pack for a coding-agent task")
    p_retrieve.add_argument("project")
    p_retrieve.add_argument("task")
    p_retrieve.add_argument("--top-k", type=positive_int, default=None, help="Number of chunks (default from config)")
    p_retrieve.add_argument("--budget", type=positive_int, default=None, help="Max estimated context tokens (default from config)")
    p_retrieve.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_metrics = sub.add_parser("metrics", help="Show cost-saving and retrieval metrics")
    p_metrics.add_argument("project")
    p_metrics.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_metrics.set_defaults(func=cmd_metrics)

    p_emb_rebuild = sub.add_parser("embeddings-rebuild", help="Rebuild FAISS index for a namespace from stored vectors")
    p_emb_rebuild.add_argument("project")
    p_emb_rebuild.add_argument("namespace")
    p_emb_rebuild.set_defaults(func=cmd_embeddings_rebuild)

    p_emb_remove = sub.add_parser("embeddings-remove", help="Remove ids from FAISS namespace")
    p_emb_remove.add_argument("project")
    p_emb_remove.add_argument("namespace")
    p_emb_remove.add_argument("ids", nargs='+')
    p_emb_remove.set_defaults(func=cmd_embeddings_remove)

    p_emb_compact = sub.add_parser("embeddings-compact", help="Compact numeric id space for a namespace")
    p_emb_compact.add_argument("project")
    p_emb_compact.add_argument("namespace")
    p_emb_compact.set_defaults(func=cmd_embeddings_compact)

    p_emb_health = sub.add_parser("embeddings-health", help="Show health info for a FAISS namespace")
    p_emb_health.add_argument("project")
    p_emb_health.add_argument("namespace")
    p_emb_health.set_defaults(func=cmd_embeddings_health)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
