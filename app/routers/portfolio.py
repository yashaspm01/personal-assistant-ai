"""
Portfolio module — auto-pulls your public GitHub repos and serves them as
structured JSON for a future frontend to render. No auth on this router on
purpose: it's meant to be public-facing, unlike Docs/Transactions/Gmail.

CUSTOM_HIGHLIGHTS lets you attach a hand-written blurb to specific repos
(by exact repo name) without losing the auto-pulled stars/language/updated
data — best of both: low maintenance, but still lets you control the pitch
for your best projects.
"""
import httpx
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.services.github_client import gh_headers

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Add entries here for repos you want a custom description on.
# Key = exact GitHub repo name. Leave empty for pure auto-pulled data.
CUSTOM_HIGHLIGHTS = {
    "personal-assistant-ai": {
        "highlights": [
            "Full RAG pipeline: chunking, embeddings, hybrid retrieval, grounded generation",
            "Swappable LLM provider (Claude/OpenAI/Groq)",
            "Multi-turn conversation memory with query rewriting",
        ],
    },
}


@router.get("/projects")
async def get_projects():
    url = f"https://api.github.com/users/{settings.github_username}/repos"
    params = {"sort": "updated", "direction": "desc", "per_page": 30}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=gh_headers(), params=params)
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"GitHub user not found: {settings.github_username}")
        resp.raise_for_status()
        repos = resp.json()

    projects = []
    for repo in repos:
        if repo.get("fork"):
            continue  # skip forked repos — only show your own original work

        custom = CUSTOM_HIGHLIGHTS.get(repo["name"], {})
        projects.append({
            "name": repo["name"],
            "description": repo.get("description") or "No description provided.",
            "url": repo["html_url"],
            "language": repo.get("language"),
            "stars": repo.get("stargazers_count", 0),
            "updated_at": repo.get("updated_at"),
            "topics": repo.get("topics", []),
            "highlights": custom.get("highlights", []),
        })

    return {"projects": projects, "source": f"github.com/{settings.github_username}"}
