# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-06-11
### Fixed
- **FAISS query off-by-one**: `query()` resolved results by list position while the
  index stores numeric ids starting at 1 — every semantic search hit returned the
  wrong chunk. Results are now resolved through the reverse idmap (works after
  removals/gaps too). Regression-tested.
- **File-lock deadlock**: `remove()`/`compact()` held the exclusive `flock` and then
  re-locked the same file via nested `_load()`/`rebuild()` calls, blocking forever
  on Linux. The lock is now re-entrant per process (per-path RLock + depth-counted
  OS lock at the outermost frame).
- **Persisted FAISS index now used at query time**: retrieval previously re-encoded
  the entire corpus on every `xkit retrieve`. Embedding/hybrid retrieval now loads
  the on-disk index and encodes only the query (O(1) model calls per search).
- **Token budget is a hard ceiling**: the first selected chunk could previously
  bypass `--budget`. Selection is now two-phase: greedy fit, then truncate the top
  chunk only if nothing fits.
- Stale vectors for changed/deleted files are removed from FAISS on `xkit update`.
- `tiktoken` encoder is cached per process instead of constructed per call.
- CLI: missing `shutil` import (NameError in export/import/cleanup), and the
  `embeddings-export/import/cleanup` subcommands are now actually registered.
- CI: tests can fail the build again (`|| true` removed); matrix covers
  Linux/macOS/Windows and minimal/full installs; ruff lint gate added.

### Added
- **BM25 retriever** (new default): dependency-free Okapi BM25 with an inverted
  index, plus camelCase/snake_case identifier splitting in the tokenizer.
- **Hybrid retrieval** (`retriever = "hybrid"`): BM25 + embeddings fused with
  Reciprocal Rank Fusion.
- **MCP server** (`xkit mcp <project>`, requires `xkit[mcp]`): exposes
  `retrieve_context`, `update_index`, `index_project`, and `get_metrics` as native
  tools for Claude Code, Cline, Cursor, and other MCP clients.
- Tree-sitter chunking via prebuilt wheels (`tree-sitter-language-pack`,
  `tree-sitter-languages`, or official `tree_sitter_<lang>` packages) — no more
  manual grammar compilation; AST chunking splits at symbol boundaries and
  line-splits oversized functions.
- `LICENSE` (MIT), `ruff.toml`, proper `.gitignore`.
- Config: `max_file_bytes` (skip generated/minified files >2 MB),
  `metrics_history_limit` (caps metrics.json growth).

### Changed
- Default retriever is `bm25` (was `tfidf`; `tfidf` remains available).
- Files larger than `max_file_bytes` are skipped instead of truncated at 50 MB.
- Indexer thread pool sizes to CPU count; embedding upsert logic deduplicated.
- Removed committed `__pycache__`, `.DS_Store`, and the stray git bundle;
  removed the no-op scheduled compaction workflow and obsolete grammar build script.
- `pyproject.toml`: modern extras (`tokens`, `treesitter`, `embeddings`, `mcp`),
  build-system block, classifiers, Python ≥3.11.

## [0.3.0] - 2026-06-09
### Added
- macOS-safe retrieval defaults and TF‑IDF fallback when embeddings may be unsafe
- `--no-embeddings` and `--allow-embeddings` CLI options for `xkit retrieve`
- `allow_embeddings` per-project config option in `.xkit/config.toml`
- Documentation and troubleshooting steps for embeddings on macOS
- Release scripts and developer docs

### Fixed
- Hardened `xkit update` against native OpenMP/MKL/tokenizers crashes on macOS by setting conservative env defaults at startup
- Multiple retrieval and CLI robustness improvements

### Tests
- Test suite updated and passing (38 passed, 3 skipped)
