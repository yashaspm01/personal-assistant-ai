"""
Thin wrapper over ChromaDB. Each module gets its own "collection" (docs,
emails, news) so retrieval never leaks context across modules by accident.
"""
import chromadb
from app.config import settings
from app.services.embedding_service import embed_texts

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection(name: str):
    return _get_client().get_or_create_collection(name=name)


def add_chunks(collection_name: str, chunks: list[dict]):
    """
    chunks: list of {id, text, source, metadata(optional dict)}
    Embeds and upserts into the given collection.
    """
    if not chunks:
        return  # nothing to embed — avoids crashing the embedding model on empty input
    collection = get_collection(collection_name)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [c["id"] for c in chunks]
    metadatas = [
        {"source": c["source"], **c.get("metadata", {})} for c in chunks
    ]
    collection.upsert(
        ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
    )


def delete_by_source(collection_name: str, source: str):
    """
    Removes all existing chunks for a given source before re-indexing it —
    makes uploads/re-indexes idempotent instead of accumulating duplicates
    every time the same file/repo gets processed again.
    """
    collection = get_collection(collection_name)
    collection.delete(where={"source": source})


def list_sources(collection_name: str) -> list[dict]:
    """Returns each unique source in a collection with its chunk count —
    powers "what's currently indexed" views in the UI."""
    collection = get_collection(collection_name)
    data = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        source = meta.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return [{"source": s, "chunks": c} for s, c in counts.items()]


def query(collection_name: str, question: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
    """
    Returns top_k most similar chunks: [{text, source, chunk_id, distance}]
    Lower distance = more similar (Chroma uses L2/cosine depending on config).
    `where` restricts the search to chunks matching metadata (e.g. one repo),
    so retrieval doesn't leak results in from unrelated indexed content.
    """
    collection = get_collection(collection_name)
    query_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)

    out = []
    if not results["ids"][0]:
        return out
    for i in range(len(results["ids"][0])):
        out.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": results["distances"][0][i],
            }
        )
    return out
