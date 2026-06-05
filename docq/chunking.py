"""Simple, fast, effective text chunking with overlap."""

from __future__ import annotations
from typing import Iterable, Tuple


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> Iterable[Tuple[str, int, int]]:
    """Yield (chunk, start_char, end_char). Overlap helps context continuity."""
    if not text:
        return
    text = text.strip()
    if len(text) <= chunk_size:
        yield text, 0, len(text)
        return

    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk, start, end
        if end == n:
            break
        start = max(end - overlap, start + 1)  # ensure progress
