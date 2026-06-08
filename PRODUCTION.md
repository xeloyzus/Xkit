# Xkit Production Notes

This document summarizes steps and considerations to run Xkit on large repositories.

Prerequisites
- Python 3.11+ (virtualenv)
- Build tools for tree-sitter grammars (gcc/clang, git, python-dev headers)
- Optional: GPU for embeddings (torch) if using large sentence-transformers models

Quick install (recommended minimal):

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .[test]
```

To enable semantic embeddings and persistent vector store (optional but recommended for accuracy):

```bash
pip install -e ".[embeddings,monitoring]"
```

Tree-sitter grammars
- Use `tools/build_treesitter.py` as a starting point, but building grammars is environment-specific.
- Alternatively, keep regex fallback; tree-sitter improves chunking quality but requires per-language grammars.

Token counting
- `tiktoken` is used when available for accurate token counts. Install via the `embeddings` extra.

Observability
- A minimal Prometheus exporter is available: call `xkit.observability.start_exporter(Path('/path/to/project'), port=8000)` to expose metrics.

Deployment notes
- Run indexing on a dedicated worker machine; indexing is CPU and I/O intensive for large repos.
- Use the incremental `xkit update` in CI/PR workflows to reindex changed files only.
- Protect the `.xkit` folder in backups and ensure it's writeable by the process.

Scaling
- For very large projects (100k+ files), consider:
  - Persistent embeddings in a vector DB (FAISS/Chroma/Qdrant)
  - Parallelizing across processes and batching embedding calls
  - Using disk-based indices to avoid keeping all embeddings in RAM

Security
- Do not expose `.xkit` endpoints publicly without authentication.
- Sanitize and review retrieved code before sending to external LLM APIs.

Quick FAISS and build notes

- To install the pinned embeddings stack in an isolated env, see `requirements-embeddings.txt` and run:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-embeddings.txt
python -m pip install -e .[test]
```

- Build tree-sitter grammars non-interactively with:

```bash
python tools/build_treesitter.py
```

- To run the Prometheus exporter for a project:

```python
from pathlib import Path
from xkit.observability import start_exporter
start_exporter(Path('/path/to/project'), port=8000)
```

CI considerations

- Running the `full` CI matrix (embeddings + tree-sitter) can be resource intensive; consider scheduling or using self-hosted runners for the `full` job.

