from __future__ import annotations

import hashlib
from pathlib import Path

from .config import XkitConfig


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_text_safe(path: Path, max_file_bytes: int = 2_000_000) -> str | None:
    """Read a file as UTF-8, returning None for unreadable or oversized files.

    Source files larger than ``max_file_bytes`` (default 2 MB) are skipped:
    they are almost always generated, vendored, or minified artifacts that
    pollute retrieval quality and inflate token estimates.
    """
    try:
        if path.stat().st_size > max_file_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def iter_project_files(project_root: Path, config: XkitConfig):
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(project_root).parts
        if any(part in config.ignored_dirs for part in rel_parts):
            continue
        if path.suffix.lower() not in config.code_extensions and path.name.lower() != "dockerfile":
            continue
        yield path
