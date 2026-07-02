# DECISIONS

Rationale for every significant tool and method choice, and a record of
implementation decisions and deviations from the CLAUDE.md specification. The
architectural rationale is summarised here; CLAUDE.md holds the authoritative
long form.

## Tool and method choices

- **Qdrant over ChromaDB.** Named collections, metadata filtering (category,
  year, author), and configurable HNSW indexing. Local disk persistence needs
  no server for development; a server is used in the Docker deployment.
- **BGE asymmetric instruction prefix.** Queries are encoded with
  `"Represent this sentence for searching relevant passages: "`; documents are
  encoded bare. Mixing the two distributions depresses similarity and degrades
  retrieval, so the asymmetry is enforced in `embedder.py`.
- **Two-stage retrieval: RRF then cross-encoder.** Dense (top-20) and sparse
  (top-20) lists are merged with Reciprocal Rank Fusion (k=60) into a top-20.
  The cross-encoder scores only that top-20 and returns the top-5. Running the
  cross-encoder over all candidates would add latency with no quality gain.
- **Cross-encoder as document grader.** The cross-encoder score (threshold 0.5)
  replaces an LLM relevance grader, which would add 500-800ms and tokens per
  document for a task the cross-encoder does in under 50ms.
- **Model tiering: 8B grading, 70B generation.** Routing, grading, and rewriting
  run on LLaMA 3.1 8B; only the final generation runs on 70B. This cuts per-query
  API cost substantially with negligible classification-quality loss.
- **Three independent failure-mode edges.** Irrelevant documents route to web
  search; an ungrounded answer retries generation; an off-target answer rewrites
  the query. Conflating these prevents targeted recovery.
- **SqliteSaver checkpointing.** Graph state persists across API restarts,
  enabling multi-turn sessions to resume mid-graph.
- **Langfuse per-node tracing.** Node functions are wrapped with `@observe` so
  each emits its own span. The LangChain callback integration is not used because
  it does not surface node-level timing and token counts.
- **RAGAS against retrieved chunks.** Faithfulness and context precision are
  computed against the exact contexts the graph used, which is the only
  configuration that measures what the system actually did.
- **CRAG web fallback.** When all documents are filtered, Tavily provides
  context and the response source is reported honestly as `web`, not `corpus`.

## Implementation decisions and deviations from the specification

The CLAUDE.md directory structure is a guide. The following choices fill gaps it
left implicit or resolve ambiguities; none change the documented architecture.

1. **`src/meridian/db.py` added.** The corpus specification mandates a
   PostgreSQL metadata store but the structure did not enumerate a module for it.
   `db.py` holds the SQLAlchemy engine, session factory, and the `Paper` model
   (with an `ingestion_status` lifecycle: pending, fetched, parsed, indexed,
   failed) so re-runs are idempotent.

2. **`Dockerfile` and `.dockerignore` added.** The `docker-compose` `api`
   service uses `build: .`, which requires a Dockerfile. Neither file was listed
   in the structure.

3. **Iteration counter is incremented in two places, not only `rewrite_query`.**
   The node table assigns the increment to `rewrite_query`, but the edge table
   caps the generation-retry loop (`generate` to `check_hallucination`) with the
   same counter. That loop contains no other counting node, so without a second
   increment site it could loop until the LangGraph recursion limit. Following
   the architecture note that "any node that increments this counter checks it
   against the cap," `check_hallucination` increments the counter only when it
   returns `"yes"` (a retry is needed). The two increment sites map exactly to
   the two recovery loops and do not double-count: in the rewrite loop the
   hallucination grade is `"no"`, so only `rewrite_query` increments.

4. **Deterministic Qdrant point IDs via UUID5.** Qdrant point IDs must be
   unsigned integers or UUIDs, so the human-readable `{arxiv_id}_{chunk_index}`
   chunk ID is mapped to `uuid5(namespace, chunk_id)` and the original chunk ID
   is preserved in the payload. Re-ingestion overwrites the same point rather
   than creating duplicates, which is the intent of deterministic chunk IDs.

5. **Optional Qdrant server via `QDRANT_URL`.** The architecture specifies local
   disk persistence, while the Docker Compose part names a `qdrant` service. The
   client uses local disk when `QDRANT_URL` is empty and a server when it is set.
   Local development uses disk; the containerised deployment sets
   `QDRANT_URL=http://qdrant:6333`.

6. **`SqliteSaver` constructed from an explicit `sqlite3` connection.**
   `SqliteSaver.from_conn_string` is a context manager in current LangGraph
   versions, which does not fit a long-lived compiled graph. An explicit
   connection with `check_same_thread=False` is stable across versions and
   usable from the threaded server.

7. **RRF uses 1-based ranks.** The fused score is `sum(1 / (k + rank))` with
   `rank` starting at 1, the canonical RRF formulation, with k=60.

8. **Context precision without reference.** The evaluation set ships without gold
   answers, so `LLMContextPrecisionWithoutReference` is used; it judges context
   relevance against the generated answer. Faithfulness needs no reference.
   Supply `reference` values in `question_set.py` to enable reference-based
   metrics.

9. **Heavy ML imports are deferred.** `sentence-transformers`/`torch`, Docling,
   and the transformers tokenizer are imported inside the functions that
   construct their objects rather than at module top level. This keeps module
   import cheap and lets the test suite import the dependency graph and mock the
   factories without loading multi-gigabyte libraries.

10. **`-e .` in `requirements.txt` and a root `conftest.py`.** Installing
    requirements also installs the package in editable mode so `import meridian`
    resolves after the single documented install step. `conftest.py` additionally
    puts `src` on `sys.path` so `pytest` works on a fresh checkout without an
    install.

11. **arXiv query includes topic terms and a server-side date range.** Beyond the
    `cs.CL`/`cs.LG`/`cs.AI` categories, the query adds reasoning/evaluation topic
    terms and a `submittedDate` range so the fetched corpus is domain-relevant
    rather than an arbitrary category sample.

12. **Cross-session memory is a separate SQLite database, not the `SqliteSaver`
    checkpoint file.** `SqliteSaver` persists mid-graph state for resuming an
    in-flight run; it is not designed to be queried for conversation history.
    `session_memory`/`session_summaries` tables in a dedicated
    `sessions.db` (see `meridian/memory/session_store.py`) keep that concern
    separate and let conversation recall survive process restarts
    independent of graph checkpointing.

13. **Production RAGAS scoring reads Langfuse observation input/output rather
    than re-running the graph.** `scripts/score_production_traces.py` extracts
    the query, graded context, and generation directly from the `generate`
    and `grade_documents` spans that the existing `@observe`-wrapped nodes
    already capture. This scores what actually happened in production instead
    of replaying traffic, but it depends on the Langfuse SDK's trace and
    observation attribute names, which have moved across SDK versions; verify
    them against the installed `langfuse` version before relying on the
    script.

14. **HuggingFace Space deployment uploads `src/` and `pyproject.toml` alongside
    `spaces/app.py`.** The Space's `requirements.txt` pins `-e .`, so the
    `meridian` package must be present at the Space repo root for that install
    step to succeed. `scripts/deploy_space.py` uploads `app.py`,
    `requirements.txt`, and `README.md` from `spaces/` to the Space root and
    `src/` plus `pyproject.toml` from the project root, rather than deploying
    only the contents of `spaces/`.

## Not yet run

Per the project plan, the code has been implemented but not executed: no
dependencies installed, no corpus ingested, no models downloaded, no tests run,
and no Git operations performed. The smoke and verification commands in
CLAUDE.md (BGE prefix similarity, cross-encoder range, Qdrant client, SqliteSaver
import) should be run after `pip install -r requirements.txt` before relying on
the pipeline.

The Phase 3-5 additions (cross-session memory, production trace scoring,
HuggingFace Space deployment) are likewise implemented but unexecuted: no
`sessions.db` has been created, no live query has been made against the
compiled graph (`GROQ_API_KEY` is not configured), no traces exist in
Langfuse to score, and the Space has not been deployed (`HF_TOKEN` is not
set). See the exact manual commands in the final session summary.
