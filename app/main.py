"""
IMPORTANT — single-worker constraint:
ChromaDB's embedded/persistent client (used here) is NOT safe for multiple
concurrent processes writing to the same directory. Always run this with a
single worker: `uvicorn app.main:app` (no --workers flag), or with gunicorn
use `-w 1`. Scaling beyond one process requires switching to Chroma's
client-server mode (a separate chromadb server process) — out of scope for
this personal-use build, but flagged here so it isn't hit by surprise later.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logging_config import setup_logging, request_logging_middleware
from app.config import settings
from app.routers import docs, transactions, news, gmail, portfolio, github_repo

setup_logging()
logger = logging.getLogger("main")

app = FastAPI(
    title="Personal Assistant API",
    description="RAG + LLM backend: docs Q&A, transactions, gmail, github repo Q&A, news, portfolio",
    version="0.2.0",
)

app.middleware("http")(request_logging_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(docs.router)
app.include_router(transactions.router)
app.include_router(news.router)
app.include_router(gmail.router)
app.include_router(portfolio.router)
app.include_router(github_repo.router)


@app.on_event("startup")
async def on_startup():
    logger.info(f"Starting up. Allowed CORS origins: {settings.allowed_origins_list}")
    logger.warning(
        "Reminder: run with a single worker process only (see module docstring) "
        "— ChromaDB's embedded mode is not safe for concurrent workers."
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
