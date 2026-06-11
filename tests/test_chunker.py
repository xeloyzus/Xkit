"""Tests for the chunker module."""

from pathlib import Path

from xkit.chunker import Chunk, _find_symbol, chunk_file


def test_empty_file():
    """An empty file should produce no chunks."""
    result = chunk_file(Path("/project"), Path("/project/empty.py"), "", 6000, 400, 8)
    assert result == []


def test_single_function():
    """A single function should produce one chunk."""
    code = """def hello():
    print("hello")
    return 42
"""
    result = chunk_file(Path("/project"), Path("/project/test.py"), code, 6000, 400, 8)
    assert len(result) == 1
    assert result[0].symbol == "hello"
    assert result[0].kind == "symbol"
    assert result[0].start_line == 1
    assert result[0].end_line == 3


def test_two_functions():
    """Two functions exceeding min_chunk_chars should produce two chunks."""
    code = """def foo():
    x = 1
    y = 2
    z = 3
    return x + y + z

def bar():
    a = 10
    b = 20
    c = 30
    return a + b + c
"""
    # Use small limits to force splitting at the function boundary
    result = chunk_file(Path("/project"), Path("/project/test.py"), code, 500, 50, 8)
    assert len(result) == 2
    assert result[0].symbol == "foo"
    assert result[1].symbol == "bar"


def test_large_function_split():
    """A function exceeding max_chunk_chars should be split."""
    lines = ["def large():"] + [f"    x{i} = {i}" for i in range(200)]
    code = "\n".join(lines)
    result = chunk_file(Path("/project"), Path("/project/test.py"), code, 2000, 400, 4)
    assert len(result) >= 2
    # First chunk should have the symbol
    assert result[0].symbol == "large"


def test_overlap_lines():
    """Overlap lines should carry context between chunks."""
    lines = ["def fn():"] + [f"    x{i} = {i}" for i in range(100)]
    code = "\n".join(lines)
    result = chunk_file(Path("/project"), Path("/project/test.py"), code, 2000, 400, 5)
    if len(result) >= 2:
        # The second chunk should start with lines from the end of the first
        first_end_lines = result[0].text.splitlines()[-5:]
        second_start_lines = result[1].text.splitlines()[:5]
        # At least some overlap should exist
        assert any(line in second_start_lines for line in first_end_lines)


def test_find_symbol():
    """Symbol detection should work for various languages."""
    assert _find_symbol("def hello():") == "hello"
    assert _find_symbol("function greet() {") == "greet"
    assert _find_symbol("export function greet() {") == "greet"
    assert _find_symbol("class MyClass {") == "MyClass"
    assert _find_symbol("struct Point {") == "Point"
    assert _find_symbol("interface User {") == "User"
    assert _find_symbol("enum Color {") == "Color"
    assert _find_symbol("const x = () => {") == "x"
    assert _find_symbol("async function fetch() {") == "fetch"
    assert _find_symbol("export default class Router {") == "Router"
    assert _find_symbol("fn main() {") == "main"  # Rust functions now detected


def test_no_symbol_line():
    """Lines without symbols should return None."""
    assert _find_symbol("import os") is None
    assert _find_symbol("x = 1") is None
    assert _find_symbol("") is None
    assert _find_symbol("    pass") is None


def test_chunk_to_dict():
    """Chunk.to_dict() should return a serializable dict."""
    c = Chunk(
        chunk_id="test:abc123",
        file="test.py",
        kind="symbol",
        symbol="hello",
        start_line=1,
        end_line=5,
        text="def hello():\n    pass",
        hash="abc123",
        token_estimate=10,
    )
    d = c.to_dict()
    assert d["chunk_id"] == "test:abc123"
    assert d["file"] == "test.py"
    assert d["symbol"] == "hello"
    assert d["start_line"] == 1
    assert d["end_line"] == 5
    assert d["token_estimate"] == 10
