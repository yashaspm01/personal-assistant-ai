from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel

from app.services.auth import require_api_key
from app.services.doc_loader import extract_text
from app.services.chunking import make_chunk_records
from app.services.vector_store import add_chunks, delete_by_source, list_sources
from app.services.hybrid_search import hybrid_query
from app.services.llm_service import chat, build_rag_prompt, LLMServiceError
from app.services.conversation import save_message, get_recent_history, rewrite_query_with_history, list_sessions, get_session_messages, delete_session

router = APIRouter(prefix="/docs", tags=["docs"], dependencies=[Depends(require_api_key)])

COLLECTION = "docs"
MODULE = "docs"
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max is {MAX_UPLOAD_SIZE_BYTES} bytes.")

    try:
        text = extract_text(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    delete_by_source(COLLECTION, file.filename)
    records = make_chunk_records(text, source=file.filename)
    add_chunks(COLLECTION, records)

    return {"filename": file.filename, "chunks_created": len(records), "status": "indexed"}


@router.get("/files")
async def list_files():
    return {"files": list_sources(COLLECTION)}


@router.delete("/files")
async def delete_file(filename: str):
    delete_by_source(COLLECTION, filename)
    return {"deleted": filename}


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
    provider: str | None = None
    session_id: str | None = None


@router.post("/query")
async def query_documents(req: QueryRequest):
    history = get_recent_history(req.session_id, MODULE, limit=6) if req.session_id else []
    standalone_question = rewrite_query_with_history(req.question, history, provider=req.provider)

    retrieved = hybrid_query(COLLECTION, standalone_question, top_k=req.top_k)

    if not retrieved:
        answer = "I don't have any indexed documents to answer from yet. Upload a document first."
    else:
        system_prompt, user_message = build_rag_prompt(standalone_question, retrieved)
        try:
            answer = chat(system_prompt, user_message, provider=req.provider)
        except LLMServiceError as e:
            raise HTTPException(status_code=503, detail=str(e))

    if req.session_id:
        save_message(req.session_id, MODULE, "user", req.question)
        save_message(req.session_id, MODULE, "assistant", answer)

    return {
        "answer": answer,
        "standalone_question": standalone_question,
        "sources": [{"source": r["source"], "chunk_id": r["chunk_id"]} for r in retrieved],
    }
