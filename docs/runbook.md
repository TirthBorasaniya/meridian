# Ingestion runbook

## Daily run
The `scheduler` service (local stack) or `.github/workflows/daily-index.yml` (CI) runs
`meridian ingest --since last_success` once a day.

1. **Candidates.** Newer versions of corpus papers come first, then new submissions since the
   last accepted watermark minus three days, oldest first, up to `max_papers_per_run`.
   Versions already indexed, withdrawn, superseded, or failed three times are skipped.
2. **Acquisition.** PDFs land in `data/pdfs/{id}v{ver}.pdf` with SHA-256 in
   `data/manifest.sqlite`. Parses land in `data/parsed/`; a `pypdf` fallback is flagged.
3. **Indexing.** Chunks are upserted into the collection behind `meridian_live` with the
   batch's `ingest_batch_id`.
4. **Gate.** R-fast runs through the full pipeline. If recall@5 is more than 0.02 below
   `data/gate_baseline.json`:
   - every point of the batch is deleted,
   - its papers are marked `rejected`,
   - an alert is appended to `data/alerts.jsonl`,
   - and the watermark does not move, so the next run retries the same window.

   On pass, the baseline is updated, superseded versions' points are deleted, and the
   watermark advances.
5. **Record.** Every batch writes `data/batches/{batch_id}.json` and a manifest row. CI
   appends `results/ingest_runs.jsonl`.

Exit codes of `meridian ingest`: 0 accepted, 2 rejected by the gate, 3 aborted at the deadline
(the partial batch is rolled back).

## A rejected batch
1. Read the GitHub issue or `data/alerts.jsonl` for the gate report.
2. Inspect the batch: `sqlite3 data/manifest.sqlite "select arxiv_id, version, status, error
   from papers where ingest_batch_id = '<batch>'"`.
3. Look for parse fallbacks or very short parses among the rejected papers.
4. If one paper causes the regression, mark it permanently skipped: `update papers set
   status = 'withdrawn', error = 'excluded after gate review' where arxiv_id = '<id>'`. The
   next run then proceeds. Never edit the baseline to let a batch through.

## A rebuild
1. Change `configs/index.yaml` (or pass `--index-config`), for example a new chunker or model.
2. `make rebuild` builds `meridian_v{N+1}`, runs R-full and A-full on both versions, and
   switches the alias only if the gate passes. The report is in `data/rebuilds/`.
3. Publish the new version: `make release`.

## Rollback
`make rollback` points `meridian_live` at the newest kept older version (one is kept). Then
publish it with `make release` so the demo and CI follow.

## Releases and restore
- `make release` snapshots the live collection and uploads the snapshot, manifest, baseline,
  and release manifest to `HF_INDEX_REPO`, then moves `latest.json`.
- `make restore` (and the Space at startup and hourly) restores the release named in
  `latest.json` into a fresh `meridian_v{N}_r{timestamp}` copy and moves the alias.

## Bootstrapping a new environment
`make setup && make up`, then either `make demo` (restore the latest release) or
`make backfill && make index` (rebuild from the seed ids). Then build R once
(`make retrieval-set`), review it (`make review-set`), draft references (`make references`),
and publish the first release (`make release`) before enabling the daily workflow.
