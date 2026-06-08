from pathlib import Path
import argparse
from xkit.embeddings import FAISSEmbeddingStore


def compact_all(project: str = "."):
    root = Path(project).resolve()
    faiss_dir = root / ".xkit" / "faiss"
    if not faiss_dir.exists():
        print("No FAISS directory found; nothing to compact")
        return 0
    store = FAISSEmbeddingStore(str(faiss_dir))
    for ns in faiss_dir.iterdir():
        if not ns.is_dir():
            continue
        name = ns.name
        print(f"Compacting namespace: {name}")
        try:
            store.compact(name)
        except Exception as e:
            print(f"Failed to compact {name}: {e}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project", nargs="?", default='.')
    args = ap.parse_args()
    raise SystemExit(compact_all(args.project))
