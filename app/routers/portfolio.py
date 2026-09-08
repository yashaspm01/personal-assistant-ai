from fastapi import APIRouter

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
# No auth dependency here on purpose — this is meant to be public-facing,
# unlike your personal data modules (docs/transactions/gmail).

PROJECTS = [
    {
        "name": "Personal Assistant API",
        "description": "A RAG+LLM backend combining document Q&A, transaction "
        "tracking, email search, movie recommendations, and news digest.",
        "stack": ["FastAPI", "ChromaDB", "Claude/OpenAI", "SQLite"],
        "highlights": [
            "Full RAG pipeline: chunking, embeddings, retrieval, grounded generation",
            "Swappable LLM provider (Claude/OpenAI)",
            "Deployed on AWS EC2 free tier",
        ],
    },
    # Add more of your projects here — this list is what your portfolio UI will render.
]


@router.get("/projects")
async def get_projects():
    return {"projects": PROJECTS}
