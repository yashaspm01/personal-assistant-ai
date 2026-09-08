"""
Hybrid retrieval: combines vector similarity search with BM25 keyword search
via Reciprocal Rank Fusion (RRF). Pure vector search can miss exact keyword
matches (names, IDs, acronyms — "Digitap", "CGPA") that don't embed
distinctively; pure keyword search misses semantic paraphrases. RRF combines
both rankings without needing to calibrate a threshold across two different,
non-comparable score scales (cosine/L2 distance vs. BM25 score).
"""
from rank_bm25 import BM25Okapi
from app.services.vector_store import get_collection, query as vector_query

RRF_K = 60  # standard constant from the original RRF paper — dampens the
            # influence of any single rank position


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def hybrid_query(collection_name: str, question: str, top_k: int = 5, candidate_pool: int = 20) -> list[dict]:
    collection = get_collection(collection_name)
    all_data = collection.get(include=["documents", "metadatas"])
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    if not all_ids:
        return []

    # --- Vector ranking ---
    vector_results = vector_query(collection_name, question, top_k=min(candidate_pool, len(all_ids)))
    vector_rank = {r["chunk_id"]: i for i, r in enumerate(vector_results)}

    # --- BM25 keyword ranking ---
    # Rebuilt per query — fine at personal scale (hundreds/low thousands of
    # chunks). At larger scale, cache this index and invalidate on writes.
    tokenized_corpus = [_tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(question))
    bm25_ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:candidate_pool]
    bm25_rank = {all_ids[i]: rank for rank, i in enumerate(bm25_ranked_indices)}

    # --- Reciprocal Rank Fusion ---
    candidate_ids = set(vector_rank.keys()) | set(bm25_rank.keys())
    fused_scores = {}
    for cid in candidate_ids:
        score = 0.0
        if cid in vector_rank:
            score += 1.0 / (RRF_K + vector_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[cid])
        fused_scores[cid] = score

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    id_to_index = {cid: i for i, cid in enumerate(all_ids)}
    results = []
    for cid in ranked_ids:
        idx = id_to_index[cid]
        results.append({
            "chunk_id": cid,
            "text": all_docs[idx],
            "source": all_metas[idx].get("source", "unknown"),
            "fused_score": fused_scores[cid],
        })
    return results
