"""Tests for the files module."""

import tempfile
from pathlib import Path

from xkit.config import XkitConfig
from xkit.files import iter_project_files, read_text_safe, sha256_text


def test_sha256_text():
    """SHA256 hash should be deterministic and 16 chars."""
    h1 = sha256_text("hello")
    h2 = sha256_text("hello")
    h3 = sha256_text("world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_sha256_text_empty():
    """Empty string should still produce a hash."""
    h = sha256_text("")
    assert len(h) == 16


def test_read_text_safe():
    """read_text_safe should read a text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')")
        path = Path(f.name)
    try:
        content = read_text_safe(path)
        assert content == "print('hello')"
    finally:
        path.unlink()


def test_read_text_safe_nonexistent():
    """read_text_safe should return None for missing files."""
    result = read_text_safe(Path("/nonexistent/file.py"))
    assert result is None


def test_read_text_safe_large_file():
    """Files over max_file_bytes are skipped — they're generated/minified noise."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x" * 3_000_000)  # > 2MB default cap
        path = Path(f.name)
    try:
        assert read_text_safe(path) is None
        # ...but readable when the caller raises the cap
        assert read_text_safe(path, max_file_bytes=5_000_000) is not None
    finally:
        path.unlink()

def test_iter_project_files():
    """iter_project_files should yield only code files."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Create some files
        (root / "main.py").write_text("x = 1")
        (root / "utils.js").write_text("const x = 1")
        (root / "README.md").write_text("# Docs")
        (root / "data.json").write_text('{"key": "val"}')
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("[core]")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "dep.js").write_text("module.exports = {}")

        config = XkitConfig()
        files = list(iter_project_files(root, config))
        rels = {str(f.relative_to(root)) for f in files}

        assert "main.py" in rels
        assert "utils.js" in rels
        assert "README.md" in rels
        assert "data.json" in rels
        assert ".git/config" not in rels
        assert "node_modules/dep.js" not in rels
