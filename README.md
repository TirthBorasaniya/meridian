# Meridian

Agentic retrieval-augmented generation over a domain-specific arXiv corpus of
LLM reasoning and evaluation papers. Meridian demonstrates production retrieval
patterns: automated corpus ingestion, hybrid retrieval with Reciprocal Rank
Fusion and cross-encoder reranking, a LangGraph state machine with three
independent failure-mode recovery paths, tiered LLM inference, graph
checkpointing, RAGAS evaluation against actually retrieved chunks, and
per-node observability.

## How it works

A query flows through a compiled LangGraph state machine:

```
route_query -> retrieve -> grade_documents -> generate -> check_hallucination -> check_answer -> END
                              |                                  |                      |
                              v (no doc passes)                  v (ungrounded)         v (off-target)
                          web_search ----> generate          generate            rewrite_query -> retrieve
```

- **Retrieve.** Dense retrieval (Qdrant, BGE embeddings) and sparse retrieval
  (BM25) each return 20 candidates. Reciprocal Rank Fusion (k=60) merges them.
- **Grade.** A cross-encoder (BAAI/bge-reranker-base) scores each candidate and
  filters anything below 0.5, returning the top 5. It doubles as the relevance
  grader, replacing a slower LLM grader.
- **Three independent recovery paths.** If every document is filtered, the graph
  falls back to Tavily web search (CRAG). If the answer is ungrounded, it
  retries generation. If the answer is off-target, it rewrites the query and
  re-retrieves. All loops are bounded by an iteration cap of 3.
- **Model tiering.** Routing and grading run on LLaMA 3.1 8B; only final
  generation runs on 70B.

## Stack

LangGraph, Qdrant, rank_bm25, sentence-transformers (BGE), Groq (LLaMA 3.1
8B/70B), Tavily, Langfuse, RAGAS, Prefect, Docling, PostgreSQL, FastAPI, Docker
Compose, GitHub Actions. See `CLAUDE.md` for the full stack table and
architecture rationale, and `DECISIONS.md` for the reasoning behind each choice.

## Requirements

- Python 3.11 (strictly; not 3.12 or 3.13)
- PostgreSQL 15 (or use Docker Compose)
- API keys: Groq (required), Tavily (web fallback), Langfuse (optional tracing)

## Setup

```bash
# 1. Create and activate a virtualenv
python3.11 -m venv venv && source venv/bin/activate

# 2. Install dependencies (also installs the meridian package in editable mode)
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env and set GROQ_API_KEY, TAVILY_API_KEY, DATABASE_URL, etc.
```

## Usage

```bash
# Smoke ingestion of 10 papers (fetch -> parse -> chunk -> embed -> index)
python scripts/ingest.py --max-papers 10

# Full corpus ingestion (target 150-300 papers)
python scripts/ingest.py

# Serve the API
uvicorn src.meridian.api.main:app --reload --host 0.0.0.0 --port 8000

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is chain-of-thought prompting?"}'

# Evaluate (20-question smoke run, then full suite)
python scripts/evaluate.py --num-questions 20
python scripts/evaluate.py

# Full stack with Docker Compose (PostgreSQL + Qdrant + API)
docker compose up --build

# Tests
pytest
```

## API endpoints

| Method | Path            | Description                                        |
|--------|-----------------|----------------------------------------------------|
| POST   | `/query`        | Run the agentic RAG graph for a query.             |
| GET    | `/health`       | Service readiness and indexed point count.         |
| GET    | `/eval-summary` | Most recent persisted RAGAS metric summary.        |

## Project layout

```
src/meridian/
  config.py            settings (pydantic-settings)
  db.py                PostgreSQL metadata models and session factory
  ingestion/           arXiv client, Docling parser, chunker, embedder, indexer, Prefect flows
  retrieval/           dense, sparse (BM25), RRF fusion, cross-encoder reranker
  graph/               state, chains, nodes, edges, compiled graph
  evaluation/          75-question set, RAGAS runner
  api/                 FastAPI app, routes, schemas
scripts/               ingest.py, evaluate.py
tests/                 retrieval, node, edge, and API tests
```

## Safety scope

`CLAUDE.md` carries a mandatory "Strict Scope Instructions" section restricting
all automated file access to this project folder. Tooling operating on this
repository must not read, write, or execute anything outside it.

## License

No license file is included yet. Add one before publishing if you intend to
permit reuse.
