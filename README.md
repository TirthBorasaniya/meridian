# Meridian

An agentic retrieval-augmented generation system over a corpus of arXiv papers
on LLM reasoning and evaluation.

## Problem

Naive RAG fails silently: when retrieval misses, the model answers confidently
from irrelevant context, and when the model hallucinates, nothing in the
pipeline notices, so a wrong answer is indistinguishable from a right one at
the API boundary. Meridian detects each failure separately and recovers
differently for each: irrelevant retrieval falls back to web search, an
ungrounded answer retries generation against the same context, and an
on-topic-but-off-target answer rewrites the query and retrieves again.

Most tutorial implementations collapse these three into one check, which
forces one recovery for three different faults. A hallucinated answer drawn
from good documents should retry generation, not discard the documents and
search the web; a well-grounded answer that misses the question should rewrite
the query, not re-generate the same text. Keeping the edges independent is what
makes targeted recovery possible.

```
route_query -> retrieve -> grade_documents -> generate -> check_hallucination -> check_answer -> END
                              |                                  |                      |
                              v (no doc passes)                  v (ungrounded)         v (off-target)
                          web_search ----> generate          generate            rewrite_query -> retrieve
```

## Results

RAGAS evaluation over 199 papers and 10,997 chunks, scored against the chunks
the retriever actually returned rather than the full corpus.

| Metric | Score | Samples scored |
|---|---|---|
| Faithfulness | 0.9285 | 14 / 20 |
| Context precision (no reference) | 0.7864 | 17 / 20 |
| Answer relevancy | 0.9035 | 20 / 20 |

Run configuration:

- **Question set:** 20 questions, the smoke subset of the 75-question set.
- **Judge:** `openai/gpt-oss-120b` served by Groq, `strictness=1`,
  `max_tokens=4096`, `reasoning_effort="low"`.
- **Generator:** `llama-3.3-70b-versatile`, also served by Groq.
- **Graph behaviour:** 19 of 20 questions were answered from the corpus, 1 fell
  back to web search, and 17 needed no recovery iteration at all.

**Coverage is incomplete, and the means above are computed over the scored
subset, not over all 20 questions.** Nine judge jobs failed during the run:
seven hit Groq's free-tier daily token cap (200,000 tokens per day) and two
timed out. Faithfulness is therefore an average over 14 samples and context
precision over 17. The lost jobs skewed toward the token-heaviest requests, so
they are not a random sample of the question set, and the direction of that
bias is unknown. A clean run needs a judge budget that covers the whole suite.

**The judge shares an inference provider with the generator.** Both run on
Groq. The judge is at least a different model family from the generator, so
faithfulness is not self-assessed, but a judge behind a fully independent
provider would be a stronger check and is the configuration to prefer. Set
`RAGAS_JUDGE_PROVIDER` to `openai` or `anthropic` to get one.

Against the project's targets, faithfulness (>= 0.87) is met and context
precision (>= 0.81) is not. Results are written to
`data/eval_results/latest.json` and served at `GET /eval-summary`; each
artifact records the judge identity and per-metric scored counts alongside the
scores, because a RAGAS number is not interpretable without knowing which model
produced it and how many samples it covers.

### Operational characteristics

Ingesting 199 papers took **1 hour 14 minutes**, of which roughly 67 minutes
was Docling PDF parsing. Parsing, not embedding or network I/O, is the
bottleneck, and it scales with PDF size rather than page count: the corpus
contains PDFs up to 37 MB, and a single 31 MB paper takes up to two minutes on
its own. Budget ingestion time from the size distribution of the corpus, not
from the paper count.

One paper of 200 was excluded. `2309.12481v2` is a withdrawn arXiv version, for
which arXiv serves no PDF, so the pipeline skips it before issuing a request.

## Quick start

Requires Python 3.11 strictly (not 3.12 or 3.13), PostgreSQL 15 (or Docker
Compose), a Groq API key, and optionally Tavily for the web fallback and
Langfuse for tracing.

```bash
# 1. Environment
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configuration
cp .env.example .env
# edit .env: GROQ_API_KEY, TAVILY_API_KEY, DATABASE_URL, QDRANT_URL

# 3. Back the stack with Postgres and Qdrant
docker compose up -d postgres qdrant

# 4. Ingest a 10-paper smoke corpus, then the full corpus
python scripts/ingest.py --max-papers 10
python scripts/ingest.py

# 5. Serve
uvicorn src.meridian.api.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is chain-of-thought prompting?", "session_id": "demo"}'

# Evaluate: 20-question smoke run first, then the full 75-question suite
python scripts/evaluate.py --num-questions 20
python scripts/evaluate.py

# Score sampled production traces from Langfuse
python scripts/score_production_traces.py --n 50

# Full stack, and tests
docker compose up --build
pytest
```

| Method | Path | Description |
|---|---|---|
| POST | `/query` | Run the agentic RAG graph for a query. |
| GET | `/health` | Service readiness and indexed point count. |
| GET | `/eval-summary` | Most recent persisted RAGAS metric summary. |

## Limitations

- **Single-domain corpus.** The corpus is arXiv papers on LLM reasoning and
  evaluation. That domain was chosen deliberately, because every question is
  answerable from a specific passage and a hallucination is therefore
  detectable, but it also means the reported scores say nothing about how the
  system behaves on heterogeneous or open-domain content.
- **No real users.** Every number here comes from a fixed question set that was
  written alongside the system. There is no production traffic, so there is no
  evidence about the queries real users would actually ask, and the
  question set may well be easier and better-matched to the corpus than real
  usage would be.
- **Recovery is bounded at three iterations.** The cap prevents unbounded
  loops, but it also means a query needing more than three rounds of rewriting
  or regeneration returns the best answer available at the cap rather than a
  correct one. One of the 20 evaluation questions hit the cap.
- **The judge shares an inference provider with the generator.** Both run on
  Groq. This is weaker than a judge behind an independent provider, and the
  same provider's daily token cap is what truncated the evaluation run.
- **One paper is excluded from the corpus.** `2309.12481v2` is a withdrawn
  arXiv version. Falling back to its earlier version was possible and was
  rejected: the paper was withdrawn because it "contains erroneous
  evaluations", and a corpus meant to support grounded answers about
  evaluation methodology should not serve retracted claims as sourced fact.

## Architecture

### The cross-encoder reranker doubles as the document grader

`grade_documents` does not call an LLM. The same cross-encoder that reranks
candidates (`BAAI/bge-reranker-base`) also decides relevance: it scores each
post-fusion candidate against the query, filters anything below 0.5, and
returns the top 5. If nothing clears the threshold, the graph routes to the
web-search fallback.

The alternative is the conventional CRAG design, an LLM grader called once per
document. That costs 500 to 800 ms and a set of tokens per document, against
under 50 ms for the cross-encoder, and on a 20-candidate list the difference is
the dominant term in query latency. The cross-encoder is also more consistent,
because it emits a calibrated score rather than a sampled yes/no.

The trade is real and worth naming: a cross-encoder scores topical similarity,
so it cannot express the kind of judgement an LLM grader can, such as "this
passage is about the right subject but answers a different question". Grading
nuance is exchanged for latency and consistency. The 0.5 threshold is the whole
of the policy, which makes it a single tunable knob rather than a prompt.
Context precision at 0.7864 is the one metric below target, and it is the
metric where that missing nuance would show up first, though the run's
incomplete coverage means this is a hypothesis rather than a measured cause.

### Model tiering: 8B for routing and grading, 70B for generation

Four of the five LLM call sites in the graph are classification: routing the
query type, grading hallucination, grading answer relevance, and rewriting the
query. All four run on `llama-3.1-8b-instant`. Only generation runs on
`llama-3.3-70b-versatile`.

Classification tasks are where a small model loses least. Each returns a
structured verdict over a short input, and the 8B model produces those about as
reliably as the 70B model does, while generation is the one step that needs the
larger context window and the stronger reasoning to synthesise five retrieved
chunks into a grounded answer. Routing only that call to the 70B model cuts
per-query API cost by roughly 60 to 70 percent.

Tiering also helps with availability, because Groq's rate limits are applied
per model: grading traffic draws on the 8B model's budget and does not consume
the generation model's. The 70B tier is the constrained one, and keeping four of
five call sites off it is what keeps a query affordable.

### Stack

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

### Corpus and retrieval

LLM reasoning and evaluation papers from arXiv (`cs.CL`, `cs.LG`, `cs.AI`),
published 2022 to 2024. Chunking is recursive character splitting at 512 tokens
with 64 tokens of overlap. The retrieval and graph layers are corpus-agnostic:
pointing the ingestion pipeline at a different document set adapts the system
to another domain.

- **Two-stage retrieval, RRF then cross-encoder.** Dense top-20 (Qdrant, BGE)
  and sparse top-20 (BM25) are independent ranked lists merged by Reciprocal
  Rank Fusion at k=60. The cross-encoder scores only that merged top-20.
  Scoring all 40 candidates would add latency for no quality gain, since RRF
  already ranked the bottom half low for good reason.
- **BGE asymmetric instruction prefix.** Queries are encoded with
  `"Represent this sentence for searching relevant passages: "`; documents are
  encoded bare. Encoding queries without the prefix draws them from a different
  distribution than the documents, which depresses similarity and measurably
  degrades retrieval.
- **Qdrant over ChromaDB.** Named collections, metadata filtering by category,
  year, and author, and configurable HNSW indexing, none of which ChromaDB
  exposes. Local disk persistence needs no server process for development.
- **Deterministic point IDs via UUID5.** Qdrant point IDs must be unsigned
  integers or UUIDs, so the readable `{arxiv_id}_{chunk_index}` chunk ID is
  mapped through `uuid5(namespace, chunk_id)` with the original preserved in
  the payload. This is what makes re-ingestion idempotent: a re-run overwrites
  the same point instead of appending a duplicate.
- **Withdrawn arXiv versions are skipped, and a 404 is never retried.** The
  arXiv API returns a paper's latest version, and arXiv serves no PDF for a
  withdrawn one. The withdrawal is detected from the submission comment that
  arrives with the metadata, so no request is issued at all.

### Graph and state

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

- **Iteration cap of 3.** The counter increments in two places,
  `rewrite_query` and `check_hallucination` (the latter only when it reports an
  ungrounded answer), because each recovery loop needs its own counting site. A
  fresh query resets the counter, so a reused `thread_id` does not inherit a
  saturated count and lose access to recovery.
- **SqliteSaver checkpointing.** Graph state persists across API restarts, so a
  long multi-turn session can resume mid-graph. The saver is built from an
  explicit `sqlite3.connect` rather than `SqliteSaver.from_conn_string`, which
  is a context manager in current LangGraph versions and does not fit a
  long-lived compiled graph.
- **Cross-session memory is a separate database.** `sessions.db` holds
  conversation turns keyed by `session_id`, distinct from the checkpointer:
  `SqliteSaver` resumes mid-graph state within a process, while session memory
  recalls prior turns across restarts and is prepended to the generation prompt.
- **CRAG web fallback reported honestly.** When every document is filtered,
  Tavily supplies the context and the response `source` is reported as `web`,
  not `corpus`. A web-retrieved answer is never presented as though it came
  from the paper corpus.

### Evaluation and observability

- **RAGAS against actually retrieved chunks.** Metrics are computed against the
  exact contexts the graph used. Scoring against the full corpus would measure
  whether an answer is consistent with some document that exists, not whether
  it is grounded in what was retrieved.
- **Context precision is measured without a reference.** The question set ships
  without gold answers, so `LLMContextPrecisionWithoutReference` judges
  retrieved context against the generated answer rather than a known correct
  one. Supplying `reference` values in `question_set.py` enables the
  reference-based metrics.
- **Langfuse per-node tracing.** Nodes are wrapped with `@observe` so each
  emits its own span with timing and token counts. The LangChain callback
  integration is deliberately not used because it does not surface node-level
  detail, which is the data that makes retrieval latency tractable to optimise.
- **Production scoring reads traces, it does not replay them.**
  `scripts/score_production_traces.py` extracts the query, graded context, and
  generation from spans the `@observe`-wrapped nodes already emit, so it scores
  what actually happened. It depends on Langfuse SDK attribute names, which have
  moved across versions; verify them against the installed version first.
- **Heavy ML imports are deferred.** `sentence-transformers`/`torch`, Docling,
  and the transformers tokenizer are imported inside the functions that
  construct their objects. Module import stays cheap, and the test suite imports
  the full dependency graph and mocks the factories without loading
  multi-gigabyte libraries.

## Project layout

```
src/meridian/
  config.py            settings (pydantic-settings)
  db.py                PostgreSQL metadata models and session factory
  ingestion/           arXiv client, Docling parser, chunker, embedder, indexer, Prefect flows
  retrieval/           dense, sparse (BM25), RRF fusion, cross-encoder reranker
  graph/               state, chains, nodes, edges, compiled graph
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
