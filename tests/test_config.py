"""Tests for the config module."""

import tempfile
from pathlib import Path

from xkit.config import XkitConfig, xkit_dir


def test_default_config():
    """Default config should have sensible values."""
    cfg = XkitConfig()
    assert cfg.index_dir_name == ".xkit"
    assert cfg.max_chunk_chars == 6000
    assert cfg.min_chunk_chars == 400
    assert cfg.overlap_lines == 8
    assert cfg.default_top_k == 8
    assert cfg.default_budget_tokens == 12000
    assert ".git" in cfg.ignored_dirs
    assert ".py" in cfg.code_extensions
    assert cfg.retriever == "bm25"


def test_xkit_dir():
    """xkit_dir should return the correct path."""
    cfg = XkitConfig()
    result = xkit_dir(Path("/project"), cfg)
    assert result == Path("/project/.xkit")


def test_xkit_dir_default():
    """xkit_dir should work without explicit config."""
    result = xkit_dir(Path("/project"))
    assert result == Path("/project/.xkit")


def test_config_load_no_file():
    """Loading config when no file exists should return defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = XkitConfig.load(Path(tmp))
        assert cfg.max_chunk_chars == 6000


def test_config_save_and_load():
    """Saving default config and loading it should work."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        default_cfg = XkitConfig()
        path = default_cfg.save_default(root)
        assert path.exists()
        assert path.parent.name == ".xkit"

        # Load it back
        loaded = XkitConfig.load(root)
        assert loaded.max_chunk_chars == 6000
        assert loaded.retriever == "bm25"


def test_config_merge():
    """Merging a partial TOML dict should override only specified fields."""
    merged = XkitConfig._merge({"max_chunk_chars": 3000, "retriever": "embeddings"})
    assert merged.max_chunk_chars == 3000
    assert merged.retriever == "embeddings"
    # Unspecified fields should keep defaults
    assert merged.min_chunk_chars == 400
    assert merged.default_top_k == 8


def test_config_merge_sets():
    """Merging lists into set fields should work."""
    merged = XkitConfig._merge({"ignored_dirs": [".git", "custom_dir"]})
    assert merged.ignored_dirs == {".git", "custom_dir"}
