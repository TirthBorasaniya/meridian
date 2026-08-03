# Meridian

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
- **Model tiering.** Routing, grading, and rewriting run on
  `llama-3.1-8b-instant`; only final generation runs on
  `llama-3.3-70b-versatile`.
- **Cross-session memory.** `meridian.memory.session_store` persists
  conversation turns to a dedicated SQLite database, keyed by `session_id`.
  This is distinct from the `SqliteSaver` checkpointer, which resumes
  mid-graph state within a running process; session memory recalls prior
  turns across process restarts and is prepended to the generation prompt as
  conversation history.

## Stack

| Layer | Tool | Notes |
|---|---|---|
| Graph orchestration | LangGraph | State graph, conditional edges, streaming |
| Vector store | Qdrant | Local disk persistence, or a server via `QDRANT_URL` |
| Sparse retrieval | rank_bm25 | BM25Okapi over the tokenised chunk corpus |
| Embeddings | BAAI/bge-small-en-v1.5 | via sentence-transformers; asymmetric query prefix |
| Reranking | BAAI/bge-reranker-base | Cross-encoder, applied post-RRF only |
| LLM inference | Groq | `llama-3.1-8b-instant` (routing, grading, rewrite), `llama-3.3-70b-versatile` (generation) |
| Observability | Langfuse | Per-node tracing via `@observe` |
| Web fallback | Tavily | CRAG fallback for out-of-corpus queries |
| Checkpointing | SqliteSaver (LangGraph) | Mid-session graph state persistence |
| Cross-session memory | SQLite (`sessions.db`) | Conversation recall across process restarts |
| Evaluation | RAGAS | Scored against actually retrieved chunks |
| Ingestion pipeline | Prefect 2.x | arXiv fetch and Docling parse flows |
| PDF parsing | Docling | Structured extraction from arXiv PDFs |
| Metadata store | PostgreSQL 15 | Paper metadata and ingestion state |
| Serving | FastAPI + Uvicorn | Lifespan context manager |
| Demo UI | Gradio | Staged under `spaces/`, not deployed |
| Infra | Docker Compose | Compose Specification format |
| CI | GitHub Actions | ruff, mypy, pytest on push |

## Architecture rationale

- **Qdrant over ChromaDB.** Named collections, metadata filtering by category,
  year, and author, and configurable HNSW indexing, none of which ChromaDB
  exposes. Local disk persistence needs no server process for development.
- **BGE asymmetric instruction prefix.** Queries are encoded with
  `"Represent this sentence for searching relevant passages: "`; documents are
  encoded bare. Encoding queries without the prefix draws them from a
  different distribution than the documents, which depresses similarity and
  measurably degrades retrieval.
- **Two-stage retrieval, RRF then cross-encoder.** Dense top-20 and sparse
  top-20 are independent ranked lists merged by Reciprocal Rank Fusion at
  k=60. The cross-encoder then scores only that merged top-20 and returns the
  top 5. Scoring all 40 candidates would add latency for no quality gain,
  since RRF already ranked the bottom half low for good reason.
- **Iteration cap of 3.** `GraphState` carries an `iteration_count`. Both
  recovery loops check it against the cap before proceeding; at the cap the
  graph returns the best available answer with a disclaimer rather than
  looping. The counter increments in two places, `rewrite_query` and
  `check_hallucination` (the latter only when it reports an ungrounded
  answer), because each recovery loop needs its own counting site. A fresh
  query resets the counter, so a reused `thread_id` does not inherit a
  saturated count.
- **SqliteSaver checkpointing.** Graph state persists across API restarts, so
  a long multi-turn session can resume mid-graph. The saver is constructed
  from an explicit `sqlite3.connect` rather than `SqliteSaver.from_conn_string`,
  which is a context manager in current LangGraph versions and does not fit a
  long-lived compiled graph.
- **Langfuse per-node tracing.** Node functions are wrapped with `@observe` so
  each emits its own span with timing and token counts. The LangChain callback
  integration is deliberately not used because it does not surface node-level
  detail, which is the data that makes retrieval latency tractable to optimise.
- **RAGAS against actually retrieved chunks.** Metrics are computed against
  the exact contexts the graph used, not the full corpus. Scoring against the
  corpus would measure whether an answer is consistent with some document that
  exists, not whether it is grounded in what was retrieved.
- **CRAG web fallback reported honestly.** When every document is filtered,
  Tavily supplies the context and the response `source` is reported as `web`,
  not `corpus`. The system does not present a web-retrieved answer as though
  it came from the paper corpus.
- **Deterministic point IDs via UUID5.** Qdrant point IDs must be unsigned
  integers or UUIDs, so the human-readable `{arxiv_id}_{chunk_index}` chunk ID
  is mapped through `uuid5(namespace, chunk_id)` and the original ID is
  preserved in the payload. This is what makes re-ingestion idempotent:
  re-running the pipeline overwrites the same point rather than appending a
  duplicate.
- **Context precision is measured without a reference.** The question set
  ships without gold answers, so `LLMContextPrecisionWithoutReference` judges
  retrieved context against the generated answer rather than against a known
  correct one. Faithfulness needs no reference either. Supplying `reference`
  values in `question_set.py` enables the reference-based metrics.
- **Production scoring reads traces, it does not replay them.**
  `scripts/score_production_traces.py` extracts the query, graded context, and
  generation from the `generate` and `grade_documents` spans that the
  `@observe`-wrapped nodes already emit, so it scores what actually happened in
  production. It depends on Langfuse SDK trace and observation attribute names,
  which have moved across SDK versions; verify them against the installed
  version before relying on it.
- **Heavy ML imports are deferred.** `sentence-transformers`/`torch`, Docling,
  and the transformers tokenizer are imported inside the functions that
  construct their objects rather than at module top level. Module import stays
  cheap, and the test suite can import the full dependency graph and mock the
  factories without loading multi-gigabyte libraries.

### Corpus

LLM reasoning and evaluation papers from arXiv (`cs.CL`, `cs.LG`, `cs.AI`),
published 2022 to 2024, targeting 150 to 300 papers. The domain is chosen so
that every question is precisely answerable from a specific paper or passage,
which is what makes faithfulness measurable: a hallucination is detectable
because the ground-truth source exists in the corpus. Chunking is recursive
character splitting at 512 tokens with 64 tokens of overlap, and chunk IDs are
deterministic (`{arxiv_id}_{chunk_index}`) so re-ingestion overwrites rather
than duplicates.

### Graph nodes

| Node | Model | Responsibility |
|---|---|---|
| `route_query` | 8B | Classify query type; set `query_type` |
| `retrieve` | — | Dense and sparse retrieval, RRF fusion |
| `grade_documents` | cross-encoder | Score each doc; filter below 0.5 |
| `web_search` | Tavily | CRAG fallback when all docs are filtered |
| `generate` | 70B | Answer from `graded_docs` or the web result |
| `check_hallucination` | 8B | Detect claims not grounded in the context |
| `check_answer` | 8B | Verify the answer resolves the original query |
| `rewrite_query` | 8B | Reformulate the query; increment `iteration_count` |

### Conditional edge routing

| After node | Condition | Route to |
|---|---|---|
| `grade_documents` | no doc scores >= 0.5 | `web_search` |
| `grade_documents` | at least one doc passes | `generate` |
| `check_hallucination` | ungrounded AND `iteration_count` < 3 | `generate` |
| `check_hallucination` | grounded | `check_answer` |
| `check_hallucination` | `iteration_count` >= 3 | END |
| `check_answer` | on target | END |
| `check_answer` | off target AND `iteration_count` < 3 | `rewrite_query` |
| `check_answer` | `iteration_count` >= 3 | END |
| `rewrite_query` | always | `retrieve` |

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

Target metrics: faithfulness >= 0.87, context precision >=
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
spaces/                Gradio demo app and Space card (staged, not deployed)
tests/                 retrieval, node, edge, state reset, ingestion, session memory, API
Dockerfile             image for the docker-compose `api` service
docker-compose.yml     api, qdrant, and postgres services
```

## Safety scope

All automated file access is restricted to this project folder. Tooling
operating on this repository must not read, write, or execute anything outside
it.

## License

MIT. See [LICENSE](LICENSE).

The licence covers this source code only. It does not cover the arXiv papers
the ingestion pipeline downloads: those remain under their respective authors'
licences, and parsed paper text is deliberately not committed to this
repository.
