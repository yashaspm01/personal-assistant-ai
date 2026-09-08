import json
import re
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.auth import require_api_key
from app.services.llm_service import chat, LLMServiceError
from app.models.db import SessionLocal, Transaction, init_db

router = APIRouter(
    prefix="/transactions", tags=["transactions"], dependencies=[Depends(require_api_key)]
)

init_db()


class IngestRequest(BaseModel):
    raw_text: str


PARSE_SYSTEM_PROMPT = """You convert messy pasted transaction text (CSV lines or
freeform lines) into a strict JSON array. Each item must have exactly these keys:
"date" (YYYY-MM-DD string), "description" (string), "amount" (number, negative
for spend, positive for income), "category" (string, your best guess if not given).
Return ONLY the JSON array, no other text, no markdown fences."""


def _strip_markdown_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json fences despite instructions not to —
    strip them defensively rather than failing the whole parse."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


@router.post("/ingest")
async def ingest_transactions(req: IngestRequest):
    """
    Uses the LLM as a flexible parser, then validates each row individually —
    one malformed row (bad date, missing field) is skipped and reported,
    not allowed to fail the entire batch.
    """
    try:
        raw_response = chat(PARSE_SYSTEM_PROMPT, req.raw_text, max_tokens=2000)
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
                date=parsed_date, description=description, amount=amount, category=category,
            ))
            created += 1
        db.commit()
    finally:
        db.close()

    return {"transactions_added": created, "rows_skipped": skipped}


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
async def query_transactions(req: QueryRequest):
    """
    Loads all transactions, gives them to the LLM as context alongside the
    question. Simpler and more reliable than text-to-SQL at personal scale;
    revisit with text-to-SQL if row count grows into the thousands.
    """
    db = SessionLocal()
    try:
        rows = db.query(Transaction).all()
    finally:
        db.close()

    if not rows:
        return {"answer": "No transactions recorded yet. Add some via /transactions/ingest first."}

    table_text = "\n".join(
        f"{r.date} | {r.description} | {r.amount} | {r.category}" for r in rows
    )
    system_prompt = (
        "You are a personal finance assistant. Answer the question using ONLY "
        "the transaction data below (format: date | description | amount | category). "
        "Negative amounts are spending, positive are income. Show your reasoning "
        "briefly (e.g. which transactions you summed)."
    )
    user_message = f"Transactions:\n{table_text}\n\nQuestion: {req.question}"

    try:
        answer = chat(system_prompt, user_message)
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"answer": answer, "transactions_considered": len(rows)}
