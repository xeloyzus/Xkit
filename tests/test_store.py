"""Tests for the store module."""

import tempfile
from pathlib import Path

from xkit.store import append_jsonl, read_json, read_jsonl, write_json, write_jsonl


def test_write_read_json():
    """Writing and reading JSON should round-trip."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.json"
        data = {"key": "value", "num": 42}
        write_json(path, data)
        loaded = read_json(path, None)
        assert loaded == data


def test_read_json_missing():
    """Reading a missing JSON file should return the default."""
    result = read_json(Path("/nonexistent/data.json"), "default")
    assert result == "default"


def test_write_read_jsonl():
    """Writing and reading JSONL should round-trip."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        rows = [{"a": 1}, {"b": 2}, {"c": 3}]
        write_jsonl(path, rows)
        loaded = read_jsonl(path)
        assert loaded == rows


def test_read_jsonl_missing():
    """Reading a missing JSONL file should return empty list."""
    result = read_jsonl(Path("/nonexistent/data.jsonl"))
    assert result == []


def test_append_jsonl():
    """Appending to JSONL should add rows."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})
        rows = read_jsonl(path)
        assert len(rows) == 2
        assert rows[0] == {"a": 1}
        assert rows[1] == {"b": 2}
