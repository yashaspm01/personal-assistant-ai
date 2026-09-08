"""
Embedding service. "local" uses sentence-transformers (free, runs on CPU,
no API cost) — good default for a 2-day free-tier build. "openai" is available
if you want higher quality embeddings and don't mind the (small) cost.
"""
from app.config import settings

_local_model = None
_openai_client = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: small, fast, free, good enough for personal RAG
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Returns one embedding vector per input text."""
    if settings.embedding_provider == "local":
        model = _get_local_model()
        vectors = model.encode(texts)
        return vectors.tolist()

    elif settings.embedding_provider == "openai":
        client = _get_openai()
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]

    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
