"""
Fixed-size chunking with overlap — the simplest, most common chunking
strategy and a good place to start learning. Overlap prevents an answer-bearing
sentence from being awkwardly split across two chunks with no shared context.
"""
import uuid


def chunk_text(
    text: str, chunk_size: int = 800, overlap: int = 150
) -> list[str]:
    """
    chunk_size / overlap are in characters (simple + provider-agnostic).
    A more advanced version could chunk by tokens or by semantic boundaries
    (paragraphs, headers) — worth trying once this baseline works.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def make_chunk_records(text: str, source: str, chunk_size=800, overlap=150) -> list[dict]:
    """Wraps chunk_text output into the {id, text, source} shape vector_store expects."""
    pieces = chunk_text(text, chunk_size, overlap)
    return [
        {"id": f"{source}-{uuid.uuid4().hex[:8]}-{i}", "text": p, "source": source}
        for i, p in enumerate(pieces)
    ]
