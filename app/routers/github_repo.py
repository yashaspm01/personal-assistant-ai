import base64
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.auth import require_api_key
from app.services.chunking import make_chunk_records
from app.services.vector_store import add_chunks, delete_by_source
from app.services.hybrid_search import hybrid_query
from app.services.llm_service import chat, build_rag_prompt, LLMServiceError
from app.services.conversation import save_message, get_recent_history, rewrite_query_with_history, list_sessions, get_session_messages, delete_session
from app.config import settings
from app.services.github_client import gh_headers

router = APIRouter(prefix="/github", tags=["github"], dependencies=[Depends(require_api_key)])

COLLECTION = "github"
MODULE = "github"

INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".json", ".yaml", ".yml", ".txt",
}
MAX_FILE_SIZE_BYTES = 100_000
MAX_FILES_TO_INDEX = 60


@router.get("/repos")
async def list_repos():
    url = f"https://api.github.com/users/{settings.github_username}/repos"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=gh_headers(), params={"per_page": 100})
        resp.raise_for_status()
        repos = resp.json()

    return {
        "repos": [
            {"name": r["name"], "default_branch": r.get("default_branch", "main")}
            for r in repos
            if not r.get("fork")
        ]
    }


class IndexRequest(BaseModel):
    repo: str
    branch: str | None = None


@router.post("/index")
async def index_repo(req: IndexRequest):
    owner = settings.github_username
    branch = req.branch

    async with httpx.AsyncClient() as client:
        if not branch:
            meta_resp = await client.get(
                f"https://api.github.com/repos/{owner}/{req.repo}", headers=gh_headers()
            )
            if meta_resp.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Repo not found: {owner}/{req.repo}")
            meta_resp.raise_for_status()
            branch = meta_resp.json().get("default_branch", "main")

        tree_url = f"https://api.github.com/repos/{owner}/{req.repo}/git/trees/{branch}?recursive=1"
        tree_resp = await client.get(tree_url, headers=gh_headers())
        if tree_resp.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Repo/branch not found or not accessible: {owner}/{req.repo}@{branch}",
            )
        tree_resp.raise_for_status()
        tree = tree_resp.json().get("tree", [])

        candidate_files = [
            item for item in tree
            if item["type"] == "blob"
            and any(item["path"].endswith(ext) for ext in INDEXABLE_EXTENSIONS)
            and item.get("size", 0) <= MAX_FILE_SIZE_BYTES
        ][:MAX_FILES_TO_INDEX]

        total_chunks = 0
        indexed_files = []
        for item in candidate_files:
            blob_resp = await client.get(
                f"https://api.github.com/repos/{owner}/{req.repo}/git/blobs/{item['sha']}",
                headers=gh_headers(),
            )
            if blob_resp.status_code != 200:
                continue
            blob = blob_resp.json()
            if blob.get("encoding") != "base64":
                continue
            try:
                content = base64.b64decode(blob["content"]).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if not content.strip():
                continue

            source_label = f"{owner}/{req.repo}/{item['path']}"
            delete_by_source(COLLECTION, source_label)
            records = make_chunk_records(content, source=source_label)
            if not records:
                continue
            for r in records:
                r["metadata"] = {"repo": f"{owner}/{req.repo}"}
            add_chunks(COLLECTION, records)
            total_chunks += len(records)
            indexed_files.append(item["path"])

    return {
        "repo": f"{owner}/{req.repo}",
        "branch": branch,
        "files_indexed": len(indexed_files),
        "chunks_created": total_chunks,
        "file_list": indexed_files,
    }


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
    repo: str
    top_k: int = 5
    session_id: str | None = None


@router.post("/query")
async def query_repo(req: QueryRequest):
    repo_filter = {"repo": f"{settings.github_username}/{req.repo}"}
    history = get_recent_history(req.session_id, MODULE, limit=6) if req.session_id else []
    standalone_question = rewrite_query_with_history(req.question, history)

    retrieved = hybrid_query(COLLECTION, standalone_question, top_k=req.top_k, where=repo_filter)
    if not retrieved:
        answer = "This repo hasn't been indexed yet (or has no matching content). Click Index above first."
    else:
        system_prompt, user_message = build_rag_prompt(standalone_question, retrieved)
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
        "sources": [{"source": r["source"]} for r in retrieved],
    }
