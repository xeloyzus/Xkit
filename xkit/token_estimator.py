from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False


@lru_cache(maxsize=2)
def _get_encoder(name: str = "cl100k_base"):
    """Load the tiktoken encoder once per process — construction is expensive."""
    return tiktoken.get_encoding(name)


def estimate_tokens(text: str) -> int:
    """Estimate tokens for `text`.

    Preferred: tiktoken (cl100k_base) for model-aligned counts; falls back to a
    character heuristic when tiktoken is not installed. Counts are estimates —
    Claude and other providers use different tokenizers, so treat budgets as
    approximate and leave headroom.
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        try:
            return max(1, len(_get_encoder().encode(text)))
        except Exception:
            pass
    # Heuristic: average ~3.7 chars per token for code-heavy text
    return max(1, int(len(text) / 3.7))
