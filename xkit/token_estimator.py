from __future__ import annotations

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False


def estimate_tokens(text: str) -> int:
    """Estimate tokens for `text`.

    Preferred: use `tiktoken` for accurate, model-aligned counts. Falls back to
    a cheap character-based heuristic if `tiktoken` is not installed.
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base (OpenAI-compatible) encoding for a reasonable default
            enc = tiktoken.get_encoding("cl100k_base")
            return max(1, len(enc.encode(text)))
        except Exception:
            # Any error falling back to heuristic
            pass
    # Heuristic: average ~3.7 chars per token for code-heavy text
    return max(1, int(len(text) / 3.7))
