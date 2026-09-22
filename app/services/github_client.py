"""
Shared GitHub API auth helper. Used by both github_repo.py (RAG Q&A over
code) and portfolio.py (public repo listing) — the only thing these two
domains actually share is "how to authenticate to GitHub's API", so that's
the only thing that lives here. Everything else stays in its own router.
"""
from app.config import settings


def gh_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers
