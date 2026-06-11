from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .config import XkitConfig, xkit_dir


def ensure_store(project_root: Path, config: XkitConfig) -> Path:
    d = xkit_dir(project_root, config)
    d.mkdir(exist_ok=True)
    return d


def write_json(path: Path, data):
    content = json.dumps(data, indent=2, sort_keys=True)
    dirpath = path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmppath).replace(path)
    finally:
        if Path(tmppath).exists():
            try:
                Path(tmppath).unlink()
            except Exception:
                pass


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]):
    dirpath = path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        Path(tmppath).replace(path)
    finally:
        if Path(tmppath).exists():
            try:
                Path(tmppath).unlink()
            except Exception:
                pass


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict):
    # Use POSIX O_APPEND to reduce risk of partial writes in concurrent scenarios.
    dirpath = path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
    try:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        fd = os.open(path, flags, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:
        # Fall back to simple append
        with path.open("a", encoding="utf-8") as f:
            f.write(line.decode("utf-8"))


def now_ms() -> int:
    return int(time.time() * 1000)
