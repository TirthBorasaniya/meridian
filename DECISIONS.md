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
