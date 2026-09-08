# Personal Assistant API

RAG + LLM backend: document Q&A, transactions, Gmail search, movie recommendations,
news digest, and a portfolio endpoint. See `PRD-personal-assistant-api.md` for full design.

## 1. Local setup (in WSL2 Ubuntu)

```bash
cd personal-assistant-api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env: add your ANTHROPIC_API_KEY and/or OPENAI_API_KEY, and APP_API_KEY (any random string)
```

Run it directly (fastest for development):
```bash
uvicorn app.main:app --reload --port 8000
```

Or via Docker (closer to how it'll run in production):
```bash
docker compose up --build
```

Visit `http://localhost:8000/docs` for interactive Swagger UI — test every
endpoint from the browser without writing a single request by hand.

## 2. Try the core RAG flow first (Doc Q&A)

```bash
# Upload a PDF
curl -X POST http://localhost:8000/docs/upload \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -F "file=@/path/to/some.pdf"

# Ask a question
curl -X POST http://localhost:8000/docs/query \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## 3. Transactions (paste text, no file needed)

```bash
curl -X POST http://localhost:8000/transactions/ingest \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "2026-09-01, Swiggy, -450, Food\n2026-09-02, Salary, +65000, Income"}'

curl -X POST http://localhost:8000/transactions/query \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "How much did I spend this week?"}'
```

## 4. Movies (needs a free TMDB API key)

Get one free at https://www.themoviedb.org/settings/api, add to `.env` as `TMDB_API_KEY`.

```bash
curl -X POST http://localhost:8000/movies/recommend \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taste_description": "slow-burn anime and 90s thrillers", "category": "anime"}'
```

## 5. News (RSS, no API key needed)

```bash
curl -X POST http://localhost:8000/news/fetch \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"topics": ["tech", "world"]}'

curl -X POST http://localhost:8000/news/query \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What happened in tech this week?"}'
```

## 6. Gmail (needs Google Cloud OAuth setup — do this first, it takes ~15-20 min)

See the docstring at the top of `app/routers/gmail.py` for the exact steps.
Also add these two packages before using it:
```bash
pip install google-auth-oauthlib google-api-python-client
```

## 7. Deployment (AWS EC2 free tier)

1. Launch a `t2.micro` EC2 instance (Ubuntu 22.04, free tier eligible)
2. SSH in, install Docker: `curl -fsSL https://get.docker.com | sh`
3. Copy this project over (`scp -r` or `git clone` if you push it to a repo)
4. Copy your real `.env` (never commit it)
5. `docker compose up -d --build`
6. Open port 8000 in the EC2 security group (or put Nginx + a domain in front later)

Your portfolio/UI (built later, on Vercel) calls this EC2 API's endpoints directly.

## Concepts implemented here (cross-reference with the PRD)

- Chunking with overlap → `app/services/chunking.py`
- Embeddings (swappable local/OpenAI) → `app/services/embedding_service.py`
- Vector storage + top-k retrieval → `app/services/vector_store.py`
- Prompt grounding + citations → `app/services/llm_service.py::build_rag_prompt`
- Tool-calling pattern (fetch external data, then reason over it) → `movies.py`
- Structured data Q&A (LLM-as-parser + LLM-as-analyst) → `transactions.py`
- OAuth2 flow → `gmail.py`
