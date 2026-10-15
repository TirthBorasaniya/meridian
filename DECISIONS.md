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
