"""
Conversation memory. Two responsibilities:
1. Persist per-session message history (SQLite — fine at personal scale).
2. Rewrite follow-up questions into standalone queries before retrieval —
   this is the actual mechanism that makes memory useful. Retrieval only
   works on the text you give it; "what about the second one?" retrieves
   nothing useful on its own, but rewritten as "What is the Finger Vein
   Authentication project?" (using history) it retrieves correctly.
"""
from app.models.db import SessionLocal, ChatMessage
from app.services.llm_service import chat, LLMServiceError


def save_message(session_id: str, module: str, role: str, content: str):
    db = SessionLocal()
    try:
        db.add(ChatMessage(session_id=session_id, module=module, role=role, content=content))
        db.commit()
    finally:
        db.close()


def get_recent_history(session_id: str, module: str, limit: int = 6) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id, ChatMessage.module == module)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
            .all()
        )
        rows.reverse()  # chronological order for the prompt
        return [{"role": r.role, "content": r.content} for r in rows]
    finally:
        db.close()


def rewrite_query_with_history(question: str, history: list[dict], provider: str | None = None) -> str:
    if not history:
        return question

    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    system_prompt = (
        "Given a conversation history and a follow-up question, rewrite the "
        "follow-up question into a standalone question containing all "
        "necessary context from the history. If it's already standalone, "
        "return it unchanged. Return ONLY the rewritten question."
    )
    user_message = f"History:\n{history_text}\n\nFollow-up question: {question}"

    try:
        rewritten = chat(system_prompt, user_message, provider=provider, max_tokens=200)
        return rewritten.strip()
    except LLMServiceError:
        # Non-critical enhancement — fall back to the original question
        # rather than failing the whole request over query rewriting.
        return question
