"""E4: seven days of simulated daily batches replayed from historical dates into a clone of the
live collection, with the real gate, plus the record of every scheduled CI run from
`results/ingest_runs.jsonl`. Writes `results/e4_ingest.json`.
"""

import json
import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

from pydantic import Field
from qdrant_client import QdrantClient, models

from meridian.common.config import VersionedConfig, load_config
from meridian.common.results import write_result

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/eval/e4.yaml"


class E4Config(VersionedConfig):
    """configs/eval/e4.yaml."""

    start_date: date
    days: int = Field(gt=0)
    max_papers_per_day: int = Field(gt=0)
    simulation_collection: str
    work_dir: str
    run_log_path: str
    o_drop_after: bool
    output: str


# ============= Collection clone =============


def clone_collection(client: QdrantClient, source: str, target: str, page_size: int = 256) -> int:
    """
    Copy every point, with vectors and payload, into a new collection with the same schema.

    Returns
    -------
    count : int
        Points copied.
    """
    from meridian.ingest import index as index_module

    metadata = index_module.collection_metadata(client, source)
    dense_dim = int(client.get_collection(source).config.params.vectors["dense"].size)  # type: ignore[index]
    if client.collection_exists(target):
        client.delete_collection(target)
    index_module.create_index_collection(
        client, target, dense_dim, {**metadata, "clone_of": source}
    )
    copied = 0
    batch_list: list[models.PointStruct] = []
    for record in index_module.iter_payloads(
        client, source, with_vectors=True, page_size=page_size
    ):
        batch_list.append(
            models.PointStruct(id=record.id, vector=record.vector, payload=record.payload)  # type: ignore[arg-type]
        )
        if len(batch_list) >= page_size:
            copied += index_module.upsert_points(client, target, batch_list, page_size)
            batch_list = []
    if batch_list:
        copied += index_module.upsert_points(client, target, batch_list, page_size)
    return copied


# ============= Run log =============


def read_run_log(path: str | Path) -> list[dict]:
    """Return every recorded scheduled run, or an empty list when none were recorded yet."""
    log_path = Path(path)
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def summarize_runs(run_list: list[dict]) -> dict:
    """Return counts of scheduled runs by decision and whether a snapshot was published."""
    decision_count: dict[str, int] = {}
    for run in run_list:
        decision_count[run.get("decision", "unknown")] = (
            decision_count.get(run.get("decision", "unknown"), 0) + 1
        )
    return {
        "runs": len(run_list),
        "by_decision": decision_count,
        "published": sum(1 for run in run_list if run.get("snapshot")),
    }


# ============= Simulation =============


def simulation_context(config: E4Config):
    """Build an ingestion context over the clone, a copied manifest, and a fresh baseline."""
    from meridian.ingest import index as index_module
    from meridian.ingest.pipeline import default_context

    live_context = default_context()
    work_dir = Path(config.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    manifest_copy = work_dir / "manifest.sqlite"
    shutil.copy2(live_context.corpus_config.paths.manifest, manifest_copy)

    from meridian.ingest.manifest import Manifest

    increment = live_context.corpus_config.arxiv.increment.model_copy(
        update={"lookback_days": 0, "o_check_versions": False}
    )
    arxiv_config = live_context.corpus_config.arxiv.model_copy(update={"increment": increment})
    corpus_config = live_context.corpus_config.model_copy(update={"arxiv": arxiv_config})
    copied = clone_collection(
        live_context.client, live_context.collection, config.simulation_collection
    )
    live_context.manifest.close()
    live_context.manifest = Manifest(manifest_copy)
    live_context.corpus_config = corpus_config
    live_context.arxiv.query_config = arxiv_config
    live_context.collection = config.simulation_collection
    live_context.baseline_path = str(work_dir / "gate_baseline.json")
    live_context.batch_dir = work_dir / "batches"
    live_context.alert_path = work_dir / "alerts.jsonl"
    live_context.search = None
    live_context.indexer = None
    logger.info("simulation collection ready", extra={"points": copied})
    return live_context, index_module.count_points(
        live_context.client, config.simulation_collection
    )


def run_e4(config: E4Config) -> dict:
    """Replay the configured days and collect the batch records and the recall trajectory."""
    from meridian.ingest.gate import run_fast_gate, save_baseline
    from meridian.ingest.pipeline import run_increment

    context, initial_points = simulation_context(config)
    initial = run_fast_gate(
        context.get_search(),
        context.gate_config,
        context.collection,
        context.baseline_path,
        context.retrieval_set_path,
    )
    save_baseline(context.baseline_path, initial, context.retrieval_set_path)
    batch_list: list[dict] = []
    trajectory = [{"day": None, "recall_at_5": initial.value, "decision": "initial"}]
    for offset in range(config.days):
        day = config.start_date + timedelta(days=offset)
        record = run_increment(
            context,
            since=day.isoformat(),
            until=day.isoformat(),
            max_papers=config.max_papers_per_day,
        )
        record_dict = record.to_dict()
        record_dict["day"] = day.isoformat()
        batch_list.append(record_dict)
        trajectory.append(
            {
                "day": day.isoformat(),
                "recall_at_5": record.eval["value"] if record.eval else None,
                "baseline": record.eval["baseline"] if record.eval else None,
                "decision": record.decision,
            }
        )
    final_points = context.client.count(context.collection, exact=True).count
    if config.o_drop_after:
        context.client.delete_collection(context.collection)
    run_list = read_run_log(config.run_log_path)
    return {
        "simulation": {
            "collection": config.simulation_collection,
            "start_date": config.start_date.isoformat(),
            "days": config.days,
            "max_papers_per_day": config.max_papers_per_day,
            "initial_points": initial_points,
            "final_points": final_points,
            "batches": [
                {
                    key: batch[key]
                    for key in (
                        "day",
                        "batch_id",
                        "papers_fetched",
                        "papers_skipped",
                        "papers_failed",
                        "papers_withdrawn",
                        "papers_indexed",
                        "chunks_added",
                        "chunks_deleted",
                        "decision",
                        "o_window_exhausted",
                        "duration_s",
                    )
                }
                for batch in batch_list
            ],
            "recall_at_5_trajectory": trajectory,
            "totals": {
                "papers_indexed": sum(batch["papers_indexed"] for batch in batch_list),
                "papers_failed": sum(batch["papers_failed"] for batch in batch_list),
                "chunks_added": sum(batch["chunks_added"] for batch in batch_list),
                "accepted": sum(1 for batch in batch_list if batch["decision"] == "accepted"),
                "rejected": sum(1 for batch in batch_list if batch["decision"] == "rejected"),
            },
        },
        "scheduled_runs": {"summary": summarize_runs(run_list), "runs": run_list},
    }


def cli_eval(args) -> int:
    """`meridian eval ingest`: run E4 and write its results file."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(E4Config, config_path)
    metrics = run_e4(config)
    write_result(
        config.output,
        metrics,
        config_path,
        notes=(
            "Simulated batches replay single historical submission days into a clone of the "
            "live collection with lookback and version checks disabled; the gate and rollback "
            "are the production code. scheduled_runs lists every CI run appended to "
            "results/ingest_runs.jsonl."
        ),
    )
    print(json.dumps(metrics["simulation"]["totals"], indent=2))
    return 0
