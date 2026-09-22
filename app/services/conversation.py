"""
Conversation memory. Three responsibilities:
1. Persist per-session message history (SQLite — fine at personal scale).
2. Rewrite follow-up questions into standalone queries before retrieval.
3. List conversations (sessions) for a "Recent" sidebar, and restore a
   full conversation's messages when one is picked — the way ChatGPT/Claude
   sidebars work (one entry per conversation, not per message).
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
        rows.reverse()
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
        return question


def list_sessions(module: str, limit: int = 15) -> list[dict]:
    """Groups messages into conversations (one entry per session_id) —
    NOT one entry per message. Title is the first user message in that
    session; sessions ordered by most recent activity."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.module == module)
            .order_by(ChatMessage.id.asc())
            .all()
        )
        sessions: dict[str, dict] = {}
        for r in rows:
            s = sessions.setdefault(
                r.session_id,
                {"session_id": r.session_id, "title": None, "last_updated": r.created_at},
            )
            if s["title"] is None and r.role == "user":
                s["title"] = r.content[:60]
            s["last_updated"] = r.created_at

        ordered = sorted(sessions.values(), key=lambda s: s["last_updated"], reverse=True)
        return ordered[:limit]
    finally:
        db.close()


def get_session_messages(module: str, session_id: str) -> list[dict]:
    """Full message history for one conversation — used to resume it."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.module == module, ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
            .all()
        )
        return [{"role": r.role, "content": r.content} for r in rows]
    finally:
        db.close()


def delete_session(module: str, session_id: str):
    """Deletes every message in one conversation — powers the 'remove'
    button on a History entry."""
    db = SessionLocal()
    try:
        db.query(ChatMessage).filter(
            ChatMessage.module == module, ChatMessage.session_id == session_id
        ).delete()
        db.commit()
    finally:
        db.close()
