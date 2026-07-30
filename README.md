# Meridian

## Live Demo

Not yet deployed. Space files are staged under `spaces/`; run
`python scripts/deploy_space.py` (requires `HF_TOKEN`) to publish to
`https://huggingface.co/spaces/TirthBorasaniya/meridian`.

Meridian is an agentic retrieval-augmented generation system: hybrid dense
and sparse retrieval fused by Reciprocal Rank Fusion, cross-encoder
reranking with CRAG-style document grading, a LangGraph state machine with
three independent failure-mode recovery paths, tiered LLM inference, graph
checkpointing, cross-session conversation memory, and RAGAS evaluation
against actually retrieved chunks, offline and in production. The reference
deployment indexes a domain-specific arXiv corpus of LLM reasoning and
evaluation papers, chosen because every question is answerable from a known
source, which makes faithfulness evaluation meaningful. The retrieval and
graph layers are corpus-agnostic: pointing the ingestion pipeline at a
different document set adapts the system to any domain.

## Features

- Hybrid retrieval: dense BGE embeddings + BM25 fused via Reciprocal Rank Fusion
- Cross-encoder reranking with CRAG-style document grading
- Query rewriting on low-relevance retrieval
- Cross-session conversation memory, persisted independently of in-session graph checkpointing
- Three independent recovery paths: retrieval fallback, hallucination retry, off-target query rewrite
- RAGAS evaluation: offline (75-question test set) and production (sampled live Langfuse traces)
- Per-node latency and token usage traced via Langfuse
- Live demo on HuggingFace Spaces (pending deployment; see above)

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
- **Grade.** `grade_documents` is the CRAG-style document grader: a
  cross-encoder (BAAI/bge-reranker-base) scores each candidate and filters
  anything below 0.5, returning the top 5. It doubles as the relevance
  grader, replacing a slower LLM grader.
- **Three independent recovery paths.** If every document is filtered, the
  graph falls back to Tavily web search (CRAG). If the answer is ungrounded,
  it retries generation. If the answer is off-target, `rewrite_query`
  reformulates the query and the graph re-retrieves. All loops are bounded by
  an iteration cap of 3.
- **Model tiering.** Routing and grading run on LLaMA 3.1 8B; only final
  generation runs on 70B.
- **Cross-session memory.** `meridian.memory.session_store` persists
  conversation turns to a dedicated SQLite database, keyed by `session_id`.
  This is distinct from the `SqliteSaver` checkpointer, which resumes
  mid-graph state within a running process; session memory recalls prior
  turns across process restarts and is prepended to the generation prompt as
  conversation history.

## Stack

LangGraph, Qdrant, rank_bm25, sentence-transformers (BGE), Groq (LLaMA 3.1
8B/70B), Tavily, Langfuse, RAGAS, Prefect, Docling, PostgreSQL, FastAPI,
Gradio, Docker Compose, GitHub Actions. See `CLAUDE.md` for the full stack
table and architecture rationale, and `DECISIONS.md` for the reasoning behind
each choice.

## Requirements

- Python 3.11 (strictly; not 3.12 or 3.13)
- PostgreSQL 15 (or use Docker Compose)
- API keys: Groq (required), Tavily (web fallback), Langfuse (optional
  tracing and production scoring), HuggingFace token (optional, Space
  deployment only)

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
  -d '{"query": "What is chain-of-thought prompting?", "session_id": "demo-session"}'

# Evaluate (20-question smoke run, then full suite)
python scripts/evaluate.py --num-questions 20
python scripts/evaluate.py

# Score sampled production traces from Langfuse (requires Langfuse credentials)
python scripts/score_production_traces.py --n 50

# Full stack with Docker Compose (PostgreSQL + Qdrant + API)
docker compose up --build

# Tests
pytest
```

## Evaluation

Target metrics (see `CLAUDE.md`): faithfulness >= 0.87, context precision >=
0.81. Results are written to `data/eval_results/latest.json` by
`scripts/evaluate.py` and served at `GET /eval-summary`.

**Status: pending.** No evaluation run has been executed against the full
corpus yet (`GROQ_API_KEY` is not configured in this environment). Populate
`.env` with a Groq API key and run:

```bash
python scripts/evaluate.py --num-questions 20   # smoke run first
python scripts/evaluate.py                       # full 75-question suite
```

This README will be updated with real faithfulness, context precision, and
answer relevancy numbers once that run has completed.

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
  graph/               state, chains, nodes (incl. document grading, query rewrite), edges, compiled graph
  memory/              cross-session conversation memory store
  evaluation/          75-question set, RAGAS runner
  api/                 FastAPI app, routes, schemas
scripts/               ingest.py, evaluate.py, score_production_traces.py, deploy_space.py
spaces/                Gradio demo app and HuggingFace Space card
tests/                 retrieval, node, edge, session memory, and API tests
```

## Safety scope

`CLAUDE.md` carries a mandatory "Strict Scope Instructions" section restricting
all automated file access to this project folder. Tooling operating on this
repository must not read, write, or execute anything outside it.

## License

MIT. See [LICENSE](LICENSE).

The licence covers this source code only. It does not cover the arXiv papers
the ingestion pipeline downloads: those remain under their respective authors'
licences, and parsed paper text is deliberately not committed to this
repository.
