"""Token counting, with a cheap fallback so chunking never hard-depends on tiktoken."""

from __future__ import annotations

from functools import lru_cache


@lru_cache
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        # ~4 characters per token is close enough for budgeting decisions.
        return max(1, len(text) // 4)
    return len(enc.encode(text))
