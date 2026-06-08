from __future__ import annotations

import hashlib
from pathlib import Path
from .config import XkitConfig


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_text_safe(path: Path) -> str | None:
    try:
        size = path.stat().st_size
        # Allow reasonably large files but avoid extremely large loads.
        if size > 50_000_000:
            # Return a truncated preview instead of skipping entirely.
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                preview = f.read(5_000_000)
            return preview + "\n\n/* file truncated due to size */\n"
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
