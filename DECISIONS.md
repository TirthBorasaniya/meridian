# Design decisions

Each entry records the context, the decision, the alternatives considered, and the
consequences. Entries are appended as the system is built and are never rewritten after the
fact; a reversal is a new entry.

## D-001: BM25 moves from an in-process index into Qdrant
Status: accepted (2026-10-14)
Context: the previous system held dense vectors in Qdrant and a separate `rank_bm25` index in
process memory, rebuilt from a full scroll of the collection at startup. A versioned index
needs both retrieval sides to change together, and an alias switch must change both at once.
Decision: each index version `meridian_v{N}` has two named vectors, `dense` (the configured
Sentence Transformers model, cosine) and `sparse` (`Qdrant/bm25` term weights from FastEmbed
with Qdrant's IDF modifier). A query runs both as prefetches (`limit=50` each) inside one
`query_points` call, fused with RRF (`k=60`, the same constant as the previous Python fusion),
and the cross-encoder reranks the fused top 30 to the final `k`.
Alternatives: keep `rank_bm25` in process (rejected: the sparse index would not be versioned
with the collection, needs a full scroll and rebuild at every start, and cannot be switched
atomically); Qdrant's default RRF constant (rejected: a different `k` would confound the
comparison with the previous path in E1).
Consequences: one collection holds both sides, so a rebuild, an alias switch, a snapshot, and
a rollback all move them together. Adding sparse vectors to the old dense-only collection was
not possible, so the first native index is a full reindex into `meridian_v1`. The previous
path is kept in `src/meridian/retrieval/legacy.py` only as the E1 baseline.

## D-002: Chunk payloads carry character spans into the parsed text
Status: accepted (2026-10-14)
Context: labeled retrieval relevance (SPEC S6.1) is defined on character spans of the parsed
paper, so that labels stay valid when the chunker changes.
Decision: the cleaned parser output is the coordinate system. Every chunk payload stores
`char_start`, `char_end`, `chunk_index`, `token_count`, and `parser_version` in addition to the
SPEC S3.4 fields. Chunk windows are cut from tokenizer offsets, so a chunk's text is exactly
`parsed_text[char_start:char_end]`.
Alternatives: label by chunk id (rejected: ids change with the chunker); fuzzy text matching
at evaluation time (rejected: ambiguous on repeated passages).
Consequences: any change to the parser's output must bump `parser.parser_version` in
`configs/index.yaml`, and labels must be rebuilt for that parser version.

## D-003: Increments advance an oldest-first watermark with a lookback and a version check
Status: accepted (2026-10-14)
Context: SPEC S3.6 runs `meridian ingest --since last_success` daily. arXiv announces papers
a day or more after submission, the recorded topic query matches several hundred submissions
a day, and SPEC S3.2 requires a newer version to replace the older one's points.
Decision: an increment queries `submittedDate` from the last accepted watermark minus
`lookback_days` (3), sorted ascending, and processes at most `max_papers_per_run` new
versions. The watermark advances to the last processed submission, or to the window end when
the window was exhausted; it does not move on a rejected or aborted batch. Before new
submissions, the latest versions of corpus papers are read through `id_list`, and newer
versions are ingested first. Old-version points are deleted only after the gate accepts the
batch, so a rejected batch leaves the previous version intact. Papers referenced by R, A, or
O stay at the version their labels refer to.
Alternatives: watermark at the run time (rejected: loses papers announced late);
`lastUpdatedDate` queries for revisions (rejected: returns every revised paper in three
categories); re-versioning evaluation papers (rejected: their span labels would stop
matching and every gate would fail).
Consequences: re-reading the lookback window costs a few API pages per run, which the manifest
turns into skips. With an unnarrowed query, the capped increments fall behind real time; the
optional `increment.extra_clause` narrows the query, and choosing it is a scope decision.

## D-004: The backfill reproduces the recorded corpus by versioned id
Status: accepted (2026-10-14)
Context: arXiv relevance ranking drifts, so re-running the relevance query would not return
the 199 papers that the existing evaluation questions were written against.
Decision: `meridian backfill --source seed-ids` fetches the 199 versioned ids in
`configs/corpus_seed_ids.txt` and reuses previously downloaded PDFs when they are present. The
backfill batch's watermark is the run time, so the first increment starts at the backfill
date rather than at the 2024-12-31 query ceiling.
Alternatives: re-run the query (kept as `--source query`, not the default); start increments
at the ceiling (rejected: it would pull almost two years of papers into one run).
Consequences: papers submitted between the query ceiling and the backfill date are not in the
corpus. The README states this.

## D-005: Training pairs come from non-evaluation papers, topped up beyond the corpus
Status: accepted (2026-10-14)
Context: R samples chunks stratified by paper, so most of the 199 corpus papers are referenced
by R, and SPEC S3.8 forbids any evaluation paper from contributing training text. The
remaining corpus papers may not yield 10,000 diverse pairs.
Decision: the training pool is every live chunk of a paper outside the R, A, and O exclusion
list. When it holds fewer than `min_pool_chunks`, relevance-ranked papers from the recorded
query beyond the corpus are downloaded, parsed, and chunked into `data/train_pool`, and they
are never indexed. The `pair_writer` role (`openai/gpt-oss-120b`) writes three queries per
chunk. The 500-pair dev split holds whole papers. Leakage is checked when pairs are written,
again before negatives are mined, and again before training; any evaluation paper aborts the
run. Hard negatives come from `mine_hard_negatives` (ranks 10 to 50, relative margin 0.05),
and a candidate from the positive's own section is skipped.
Alternatives: train on all corpus papers (rejected: leaks evaluation papers into training);
split dev by pair (rejected: dev queries would share papers with training queries).
Consequences: the adapted model is trained partly on papers that are not in the index. That
matches the corpus topic and keeps R a true held-out set. The model is promoted only through
the rebuild gate.

## D-006: Releases restore into a fresh copy, and answer caches key on the index revision
Status: accepted (2026-10-14)
Context: daily increments write into the live `meridian_v{N}`, so consecutive releases are
snapshots of the same version number. Restoring a snapshot over the live collection would
interrupt serving. An answer cache keyed only on `N` would also survive an increment that
deleted superseded chunks the cached answer cites.
Decision: a release is restored into `meridian_v{N}_r{YYYYMMDDHHMM}`; the alias then moves in
one request, and the previous copy of the same version is dropped. `parse_version` reads the
version through the suffix, and releases record the base name. Answer and semantic caches key
on `{N}:{snapshot or last ingest}`, so every accepted increment invalidates cached answers.
`latest.json` is uploaded after every other release file, and restore verifies each file's
SHA-256 before installing it.
Alternatives: restore in place with snapshot priority (rejected: the collection is
unavailable or inconsistent during recovery); a new version number per day (rejected: the
version number would stop meaning "built with this configuration").
Consequences: a restore briefly holds two copies. The answer cache restarts empty after every
daily release, which bounds its hit rate on the demo.

## D-007: The demo Space runs the Qdrant server binary next to the API
Status: accepted (2026-10-14)
Context: SPEC S5.2 prefers Qdrant local mode inside the Space if it supports named sparse
vectors and the RRF Query API, verified by a test in the image build. Releases, however, are
Qdrant server snapshots (SPEC S3.9), and local mode cannot restore a server snapshot.
Decision: the image build runs `space/check_local_mode.py`, which records in the image whether
local mode supports the query pattern. The container then runs the Qdrant server binary,
copied from the pinned `qdrant/qdrant:v1.19.2` image, as a second process, and restores the
latest snapshot into it at startup and hourly. Model weights are baked into the image to
shorten cold starts. Uvicorn trusts `X-Forwarded-For` only from private proxy ranges, so the
per-IP limits see the rightmost public address and a client cannot spoof its own.
Alternatives: local mode with a portable export of points instead of snapshots (rejected: a
second release format beside the one CI already verifies); a managed Qdrant cluster (rejected:
an always-on service the project does not otherwise need).
Consequences: one container, two processes, and the same snapshot format end to end. The local
mode check result is recorded but does not change the runtime.

## D-008: Groq tiers after the Llama retirement, and a Preview judge
Status: accepted (2026-10-15)
Context: Groq shut down `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` on 2026-08-16,
which served the router, rewriter, grader, and generator roles.
Decision: the fast roles (router, rewriter, grader) use `openai/gpt-oss-20b`, and the strong
roles (generator, question writer, pair writer) use `openai/gpt-oss-120b`. The judge must differ
from the generator, so it uses `qwen/qwen3.6-27b`. That is a Groq Preview model: Preview models
are intended for evaluation, can change without notice, and can be withdrawn on short notice.
If it disappears, the judge needs another non-generator model, and judgments cached under the
old model are not reused because the cache key includes the model. The gpt-oss models are
reasoning models, so their roles set `reasoning_effort: low` and `max_tokens` leaves room for
reasoning tokens ahead of the short JSON answer. The client also strips inline
`<think>...</think>` blocks from completions, because some models return reasoning in the
content, and a block left in place would break JSON parsing. The rule that no Llama model
writes training pairs stays, now with its rationale stated as the Llama license in general.
Alternatives: gpt-oss-120b as both generator and judge (rejected: SPEC S4.3 requires a judge
different from the generator, so faithfulness would be self-assessed).
Consequences:
- The question writer and the reference-answer drafter are now the same model as the
  generator. R labels and references are produced before and independently of any answer, and
  the judge is a different model, but the shared model is a limitation to report.
- The README's note that the judge shares a provider with the generator still holds.
- Token prices in `configs/models.yaml` are unconfirmed. The `qwen/qwen3.6-27b` entry is
  carried over from `qwen/qwen3-32b`, and the 20b entry is from memory. Both must be checked
  before E3, and `prices_checked_on` stays null until then.
- No reference to `qwen/qwen3.8-27b` existed in the repository, so no replacement was needed.

Amendment, 2026-10-15: the judge is now `qwen/qwen3.8-27b` (D-010), and prices are no longer
estimated. The judge statements above describe the decision as first made.

## D-009: Daily additions are narrowed to LLM reasoning, calibration, and chain-of-thought
Status: accepted (2026-10-15)
Context: the recorded topic query matched about 305 submissions a day (9,165 over the 30 days
from 2026-09-15 to 2026-10-14, minimum 114, maximum 641). At 40 papers per run, increments
would fall further behind every day (D-003).
Decision: `arxiv.increment.extra_clause` is
`abs:"large language model" AND (abs:"reasoning" OR abs:"calibration" OR abs:"chain-of-thought")`,
ANDed into the recorded query for daily increments only. The backfill query is unchanged. Daily
arXiv counts for those 30 days: median 24.5, mean 26.3, minimum 8, maximum 61, and 4 days above
40 (09-28: 61, 09-29: 55, 09-30: 54, 10-08: 44). The cap and the three-day lookback carry the
excess to the next run, so a burst clears in about two days. The measurement was a one-off
count against arXiv's date filter with requests spaced 3.2 seconds apart.
Alternatives, measured over the same 30 days: titles only, `ti:reasoning OR
ti:"chain-of-thought" OR ti:calibration` (median 24, 3 days above 40; rejected because the
corpus is defined on abstracts); that clause plus `(ti:evaluation AND abs:"language model")`
(median 30.5, 8 days above 40; kept as the fallback if evaluation-methodology coverage matters
more); `abs:"chain-of-thought" OR abs:"calibration"` (30-day total 1,189, mean 40, too close to
the cap).
Consequences: papers that never say "large language model" in the abstract, and benchmark or
evaluation-methodology papers that mention none of the three terms, are no longer added. The
corpus started broader than it grows. Papers added before this decision stay, and the R-fast
gate measures only those, so it will not detect the shift.

## D-010: Groq prices come from the models page, and the judge is qwen/qwen3.8-27b
Status: accepted (2026-10-15)
Context: remembered and carried-over prices were not acceptable for E3's cost numbers.
`https://groq.com/pricing` loads but shows no prices. `https://console.groq.com/docs/models`,
retrieved 2026-10-15, lists them. That page does not list `qwen/qwen3.6-27b`, the judge chosen
in D-008. Its only Qwen entry is the Preview model `qwen/qwen3.8-27b`.
Decision: the judge is `qwen/qwen3.8-27b`, replacing `qwen/qwen3.6-27b` everywhere. It is a
Groq Preview model: Preview models are intended for evaluation and can be discontinued at
short notice. `configs/models.yaml` records the prices per million tokens as retrieved from
the page named in `prices_source_url` on 2026-10-15: `openai/gpt-oss-20b` $0.075 input and
$0.30 output, `openai/gpt-oss-120b` $0.15 input and $0.60 output, and `qwen/qwen3.8-27b`
$0.80 input and $4.00 output. The earlier remembered 120b output price of $0.75 was wrong. A
model without a published price is null, and cost conversion raises an error naming it
instead of estimating.
Alternatives: keep `qwen/qwen3.6-27b` (rejected: not listed on the models page, so its
availability and price were unverified); `openai/gpt-oss-120b` as judge (rejected: it is the
generator, and SPEC S4.3 requires a different judge).
Consequences: judge calls are the most expensive per token of the three models, and the judge
runs on every cited sentence in evaluation, so E2 and E3 costs will be dominated by it. If the
judge is withdrawn, `configs/models.yaml` needs another non-generator model and its price.
Judgments cached under another model are not reused, because the cache key includes the model.
