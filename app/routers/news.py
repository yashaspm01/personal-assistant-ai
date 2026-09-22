import feedparser
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.auth import require_api_key
from app.services.chunking import make_chunk_records
from app.services.vector_store import add_chunks, delete_by_source
from app.services.hybrid_search import hybrid_query
from app.services.llm_service import chat, build_rag_prompt, LLMServiceError
from app.services.conversation import save_message, get_recent_history, rewrite_query_with_history, list_sessions, get_session_messages, delete_session

router = APIRouter(prefix="/news", tags=["news"], dependencies=[Depends(require_api_key)])

COLLECTION = "news"
MODULE = "news"

DEFAULT_FEEDS = {
    "tech": "https://feeds.arstechnica.com/arstechnica/index",
    "world": "http://feeds.bbci.co.uk/news/world/rss.xml",
}


class FetchRequest(BaseModel):
    topics: list[str] = ["tech", "world"]


@router.post("/fetch")
async def fetch_news(req: FetchRequest):
    total_indexed = 0
    for topic in req.topics:
        feed_url = DEFAULT_FEEDS.get(topic, topic)
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:10]:
            text = f"{entry.get('title', '')}\n{entry.get('summary', '')}"
            if not text.strip():
                continue
            source_label = entry.get("link", topic)
            delete_by_source(COLLECTION, source_label)
            records = make_chunk_records(text, source=source_label)
            if not records:
                continue
            add_chunks(COLLECTION, records)
            total_indexed += len(records)

    return {"topics_fetched": req.topics, "chunks_indexed": total_indexed}


@router.get("/sessions")
async def list_chat_sessions(limit: int = 15):
    return {"sessions": list_sessions(MODULE, limit=limit)}


@router.get("/sessions/{session_id}")
async def get_chat_session(session_id: str):
    return {"messages": get_session_messages(MODULE, session_id)}


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    delete_session(MODULE, session_id)
    return {"deleted": session_id}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str | None = None


@router.post("/query")
async def query_news(req: QueryRequest):
    history = get_recent_history(req.session_id, MODULE, limit=6) if req.session_id else []
    standalone_question = rewrite_query_with_history(req.question, history)

    retrieved = hybrid_query(COLLECTION, standalone_question, top_k=req.top_k)
    if not retrieved:
        answer = "No news indexed yet. Call /news/fetch first."
    else:
        system_prompt, user_message = build_rag_prompt(standalone_question, retrieved)
        try:
            answer = chat(system_prompt, user_message)
        except LLMServiceError as e:
            raise HTTPException(status_code=503, detail=str(e))

    if req.session_id:
        save_message(req.session_id, MODULE, "user", req.question)
        save_message(req.session_id, MODULE, "assistant", answer)

    return {"answer": answer, "standalone_question": standalone_question, "sources": [{"source": r["source"]} for r in retrieved]}
