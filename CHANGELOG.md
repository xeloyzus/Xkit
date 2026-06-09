# Changelog

All notable changes to this project will be documented in this file.

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
