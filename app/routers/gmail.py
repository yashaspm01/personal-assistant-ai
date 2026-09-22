"""Gmail module. Needs Google Cloud OAuth credentials — see prior setup steps.
credentials.json / token.json live in app/data/."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.auth import require_api_key
from app.services.chunking import make_chunk_records
from app.services.vector_store import add_chunks, delete_by_source
from app.services.hybrid_search import hybrid_query
from app.services.llm_service import chat, build_rag_prompt, LLMServiceError
from app.services.conversation import save_message, get_recent_history, rewrite_query_with_history, list_sessions, get_session_messages, delete_session

router = APIRouter(prefix="/gmail", tags=["gmail"], dependencies=[Depends(require_api_key)])

COLLECTION = "emails"
MODULE = "gmail"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _get_gmail_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.exceptions import RefreshError
    from googleapiclient.discovery import build
    import os

    creds = None
    token_path = "app/data/token.json"
    creds_path = "app/data/credentials.json"

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # Expected every ~7 days while the OAuth app is in "Testing"
                # status (Google's own limit, not a bug in this code) — the
                # old token is dead, remove it and force a fresh auth flow
                # below instead of leaving a broken token file around.
                os.remove(token_path)
                creds = None
        if not creds:
            if not os.path.exists(creds_path):
                raise HTTPException(status_code=500, detail="Missing app/data/credentials.json")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0, open_browser=False)
            except Exception as e:
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "Gmail access has expired (this happens every ~7 days while "
                        "the app is in Google's 'Testing' status). Check this endpoint's "
                        "terminal output for a Google sign-in URL, open it in your "
                        "browser, and approve access again."
                    ),
                ) from e
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


class FetchRequest(BaseModel):
    max_results: int = 20
    # "category:primary" matches what you see in Gmail's Primary tab by
    # default — without it, fetch pulls from Promotions/Social/Updates too,
    # which looks like "wrong" emails but is actually just a broader scope.
    query: str = "newer_than:7d category:primary"


@router.post("/fetch")
async def fetch_emails(req: FetchRequest):
    service = _get_gmail_service()
    results = service.users().messages().list(userId="me", q=req.query, maxResults=req.max_results).execute()
    messages = results.get("messages", [])

    total_indexed = 0
    for msg_ref in messages:
        msg = service.users().messages().get(userId="me", id=msg_ref["id"]).execute()
        snippet = msg.get("snippet", "")
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        subject = headers.get("Subject", "(no subject)")
        sender = headers.get("From", "unknown")
        date_header = headers.get("Date", "unknown date")
        # Including the real date lets the model actually answer date-scoped
        # questions ("what came in today") instead of guessing from whatever
        # happened to be retrieved.
        text = f"Date: {date_header}\nFrom: {sender}\nSubject: {subject}\n{snippet}"
        if not text.strip():
            continue

        source_label = f"{subject} ({sender})"
        delete_by_source(COLLECTION, source_label)
        records = make_chunk_records(text, source=source_label)
        if not records:
            continue
        add_chunks(COLLECTION, records)
        total_indexed += len(records)

    return {"emails_fetched": len(messages), "chunks_indexed": total_indexed}


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
async def query_emails(req: QueryRequest):
    history = get_recent_history(req.session_id, MODULE, limit=6) if req.session_id else []
    standalone_question = rewrite_query_with_history(req.question, history)

    retrieved = hybrid_query(COLLECTION, standalone_question, top_k=req.top_k)
    if not retrieved:
        answer = "No emails indexed yet. Call /gmail/fetch first."
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
