import base64
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.auth import require_api_key
from app.services.chunking import make_chunk_records
from app.services.vector_store import add_chunks, delete_by_source
from app.services.hybrid_search import hybrid_query
from app.services.llm_service import chat, build_rag_prompt, LLMServiceError
from app.services.conversation import save_message, get_recent_history, rewrite_query_with_history
from app.config import settings

router = APIRouter(prefix="/github", tags=["github"], dependencies=[Depends(require_api_key)])

COLLECTION = "github"
MODULE = "github"

INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".json", ".yaml", ".yml", ".txt",
}
MAX_FILE_SIZE_BYTES = 100_000
MAX_FILES_TO_INDEX = 60


def _gh_headers():
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


class IndexRequest(BaseModel):
    owner: str
    repo: str
    branch: str = "main"


@router.post("/index")
async def index_repo(req: IndexRequest):
    tree_url = f"https://api.github.com/repos/{req.owner}/{req.repo}/git/trees/{req.branch}?recursive=1"
    async with httpx.AsyncClient() as client:
        tree_resp = await client.get(tree_url, headers=_gh_headers())
        if tree_resp.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Repo/branch not found or not accessible: {req.owner}/{req.repo}@{req.branch}. "
                    "If private, check GITHUB_TOKEN is set in .env."
                ),
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
                f"https://api.github.com/repos/{req.owner}/{req.repo}/git/blobs/{item['sha']}",
                headers=_gh_headers(),
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

            source_label = f"{req.owner}/{req.repo}/{item['path']}"
            delete_by_source(COLLECTION, source_label)  # idempotent re-index
            records = make_chunk_records(content, source=source_label)
            if not records:
                continue
            add_chunks(COLLECTION, records)
            total_chunks += len(records)
            indexed_files.append(item["path"])

    return {
        "repo": f"{req.owner}/{req.repo}",
        "files_indexed": len(indexed_files),
        "chunks_created": total_chunks,
        "file_list": indexed_files,
    }


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str | None = None


@router.post("/query")
async def query_repo(req: QueryRequest):
    history = get_recent_history(req.session_id, MODULE, limit=6) if req.session_id else []
    standalone_question = rewrite_query_with_history(req.question, history)

    retrieved = hybrid_query(COLLECTION, standalone_question, top_k=req.top_k)
    if not retrieved:
        answer = "No repo indexed yet. Call /github/index first."
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
