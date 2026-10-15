# Architecture

## Data flow
1. **Ingestion** writes paper versions to `data/manifest.sqlite`, PDFs to `data/pdfs/`, and
   cleaned Markdown with section offsets to `data/parsed/`.
2. **Indexing** writes chunks to `meridian_v{N}`: one point per chunk, with the id
   `uuid5(NAMESPACE_URL, "{id}v{ver}:{section}:{chunk}")`, a `dense` vector, a `sparse`
   vector, and a payload with metadata and the character span.
3. **Serving** reads only through the alias `meridian_live`.
4. **Releases** snapshot the live collection with the manifest and the gate baseline to a
   private dataset repo. CI and the demo restore from it.

## Versioning
| Change | Mechanism |
|---|---|
| New papers (daily) | Upsert into the live collection under a new `ingest_batch_id`; R-fast gate; rollback deletes the batch |
| New paper version | Ingested in a batch; old version points deleted only after acceptance |
| Parser, chunker, dense or sparse model | Rebuild into `meridian_v{N+1}`; full gate; atomic alias switch; one previous version kept |
| Daily release | Snapshot of the same `meridian_v{N}`; restored into `meridian_v{N}_r{timestamp}` |

## Agent graph
`start_turn -> route -> {respond_without_search | rewrite} -> retrieve -> grade -> decide`
`decide -> {generate | rewrite (retry) | web_search | insufficient}`
`generate -> verify -> {generate (one regeneration) | finalize}`

## Caches
| Cache | Key | Store |
|---|---|---|
| LLM responses | SHA-256 of model, messages, parameters | SQLite (evaluation) or Redis (serving) |
| Query embeddings | model and SHA-256 of prefixed text | Redis or memory |
| Answers | normalized question, session-independence flag, index revision | Redis or memory |
| Semantic answers (off by default) | query embedding, filtered on index revision | Redis Stack via RedisVL |

## Metrics
`meridian_requests_total{route}`, `meridian_stage_latency_seconds{stage}`,
`meridian_retrieval_supporting_chunks`, `meridian_fallback_total{kind}`,
`meridian_cache_hits_total{cache}`, `meridian_cache_misses_total{cache}`,
`meridian_llm_tokens_total{model,direction}`, `meridian_ingest_batches_total{decision}`,
`meridian_index_points`.
