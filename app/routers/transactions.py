import json
import re
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.services.auth import require_api_key
from app.services.llm_service import chat, LLMServiceError
from app.services.doc_loader import extract_text
from app.services.ocr_loader import extract_text_from_image, is_image_file
from app.services.conversation import save_message, get_recent_history, rewrite_query_with_history, list_sessions, get_session_messages, delete_session
from app.models.db import SessionLocal, Transaction, init_db

router = APIRouter(
    prefix="/transactions", tags=["transactions"], dependencies=[Depends(require_api_key)]
)

init_db()

MODULE = "transactions"
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024


class IngestRequest(BaseModel):
    raw_text: str


PARSE_SYSTEM_PROMPT = """You convert messy pasted transaction text (CSV lines,
freeform lines, or OCR output from a screenshot) into a strict JSON array.
Each item must have exactly these keys: "date" (YYYY-MM-DD string),
"description" (string), "amount" (number, negative for spend, positive for
income), "category" (string, your best guess if not given). OCR text may
have garbled spacing or line breaks — use your best judgement to reconstruct
each transaction. Return ONLY the JSON array, no other text, no markdown fences."""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_and_store(raw_text: str, source: str) -> dict:
    try:
        raw_response = chat(PARSE_SYSTEM_PROMPT, raw_text, max_tokens=2000)
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))

    cleaned = _strip_markdown_fences(raw_response)
    try:
        rows = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse LLM output as JSON. Raw output: {raw_response[:500]}",
        )

    if not isinstance(rows, list):
        raise HTTPException(status_code=422, detail="LLM output was not a JSON array.")

    db = SessionLocal()
    created = 0
    skipped = []
    try:
        for i, row in enumerate(rows):
            try:
                parsed_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                description = str(row["description"])
                amount = float(row["amount"])
                category = row.get("category")
            except (KeyError, ValueError, TypeError) as e:
                skipped.append({"row_index": i, "row": row, "reason": str(e)})
                continue

            db.add(Transaction(
                date=parsed_date, description=description, amount=amount,
                category=category, source=source,
            ))
            created += 1
        db.commit()
    finally:
        db.close()

    return {"transactions_added": created, "rows_skipped": skipped, "source": source}


@router.post("/ingest")
async def ingest_transactions(req: IngestRequest):
    return _parse_and_store(req.raw_text, source="pasted text")


@router.post("/ingest-file")
async def ingest_transactions_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15MB).")

    if is_image_file(file.filename):
        raw_text = extract_text_from_image(file_bytes)
    else:
        try:
            raw_text = extract_text(file.filename, file_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    if not raw_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file — try a clearer screenshot or a different format.",
        )

    return _parse_and_store(raw_text, source=file.filename)


@router.get("/sources")
async def list_sources():
    db = SessionLocal()
    try:
        rows = db.query(Transaction).all()
    finally:
        db.close()
    counts: dict[str, int] = {}
    for r in rows:
        key = r.source or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return {"sources": [{"source": s, "count": c} for s, c in counts.items()]}


@router.delete("/sources")
async def delete_source(source: str):
    db = SessionLocal()
    try:
        db.query(Transaction).filter(Transaction.source == source).delete()
        db.commit()
    finally:
        db.close()
    return {"deleted_source": source}


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
    session_id: str | None = None


@router.post("/query")
async def query_transactions(req: QueryRequest):
    db = SessionLocal()
    try:
        rows = db.query(Transaction).all()
    finally:
        db.close()

    if not rows:
        return {"answer": "No transactions recorded yet. Add some via /transactions/ingest first.", "sources": []}

    history = get_recent_history(req.session_id, MODULE, limit=6) if req.session_id else []
    standalone_question = rewrite_query_with_history(req.question, history)

    table_text = "\n".join(
        f"{r.date} | {r.description} | {r.amount} | {r.category} | source: {r.source or 'unknown'}"
        for r in rows
    )
    system_prompt = (
        "You are a personal finance assistant. Answer the question using ONLY "
        "the transaction data below (format: date | description | amount | "
        "category | source). Negative amounts are spending, positive are "
        "income. The 'source' field tells you which file or paste each row "
        "came from — if the user asks about 'this file' or a specific "
        "filename, filter to matching source rows only. Show your reasoning briefly."
    )
    user_message = f"Transactions:\n{table_text}\n\nQuestion: {standalone_question}"

    try:
        answer = chat(system_prompt, user_message)
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if req.session_id:
        save_message(req.session_id, MODULE, "user", req.question)
        save_message(req.session_id, MODULE, "assistant", answer)

    return {
        "answer": answer,
        "standalone_question": standalone_question,
        "transactions_considered": len(rows),
        "sources": [],  # transactions aren't chunk-retrieved, kept for shape-compatibility with ChatPanel
    }
