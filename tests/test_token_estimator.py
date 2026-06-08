"""Tests for the token estimator module."""

from xkit.token_estimator import estimate_tokens


def test_estimate_tokens_empty():
    """Empty text should return 0."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_short():
    """Short text should return at least 1."""
    assert estimate_tokens("a") == 1


def test_estimate_tokens_approximate():
    """Token estimate should be roughly chars / 3.7."""
    text = "def hello():\n    print('hello world')\n    return 42\n"
    result = estimate_tokens(text)
    expected = max(1, int(len(text) / 3.7))
    assert result == expected


def test_estimate_tokens_longer():
    """Longer text should scale proportionally."""
    text = "x = 1\n" * 100
    result = estimate_tokens(text)
    assert result > 10
    assert result < len(text)  # Should be less than char count
