"""E5: base versus domain-adapted embedding model on R-full (held out from training by the
paper-level exclusion list), for dense-only retrieval and for the full hybrid plus rerank
pipeline, with training time and hardware and the rebuild gate's decision. Writes
`results/e5_embedder.json`.
"""

import json
import logging
from pathlib import Path

from pydantic import Field

from meridian.common.config import IndexConfig, VersionedConfig, get_corpus_config, load_config
from meridian.common.results import write_result
from meridian.eval.datasets import load_retrieval_set
from meridian.eval.retrieval_eval import evaluate_search, mode_search

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/eval/e5.yaml"
PIPELINE_MODE_DICT = {"dense_only": "dense", "hybrid_rerank": "hybrid_rerank"}


class E5Config(VersionedConfig):
    """configs/eval/e5.yaml."""

    retrieval_set_path: str
    split: str
    k: int = Field(ge=10)
    base_index_config: str
    adapted_index_config: str
    collection_prefix: str
    o_drop_after: bool
    train_report_path: str
    rebuild_report_dir: str
    output: str


def collection_with_model(client, prefix: str, dense_model: str) -> str | None:
    """Return a versioned collection whose metadata records `dense_model`, newest first."""
    from meridian.ingest import index as index_module

    for version in reversed(index_module.list_versions(client, prefix)):
        name = index_module.collection_name(prefix, version)
        if index_module.collection_metadata(client, name).get("dense_model") == dense_model:
            return name
    return None


def gate_decision(report_dir: str | Path, dense_model: str) -> dict | None:
    """Return the newest rebuild report whose candidate used the adapted model."""
    from meridian.ingest import index as index_module
    from meridian.ingest.index import get_qdrant_client

    client = get_qdrant_client()
    candidate_list = []
    for path in sorted(Path(report_dir).glob("*.json")):
        report = json.loads(path.read_text())
        name = report["candidate"]
        if client.collection_exists(name):
            model = index_module.collection_metadata(client, name).get("dense_model")
        else:
            model = report.get("dense_model")
        if model == dense_model:
            candidate_list.append(report)
    if not candidate_list:
        return None
    latest = candidate_list[-1]
    return {
        "candidate": latest["candidate"],
        "promoted": latest["o_promoted"],
        "reasons": latest["reason_list"],
        "retrieval": latest["retrieval"],
        "faithfulness": latest["faithfulness"],
    }


def ensure_collection(client, index_config: IndexConfig, name: str) -> tuple[str, bool]:
    """Return a collection built with this config's dense model, building one if needed."""
    from meridian.ingest.build import build_collection
    from meridian.ingest.index import live_version
    from meridian.ingest.manifest import Manifest

    existing = collection_with_model(
        client, index_config.collection_prefix, index_config.dense.model
    )
    if existing is not None:
        return existing, False
    if client.collection_exists(name):
        client.delete_collection(name)
    corpus_config = get_corpus_config()
    build_collection(
        client,
        Manifest(corpus_config.paths.manifest),
        index_config,
        corpus_config,
        collection=name,
        index_version=live_version(client, index_config) or 1,
    )
    return name, True


def run_e5(config: E5Config) -> dict:
    """Evaluate both models on both pipelines."""
    from meridian.ingest.index import get_qdrant_client
    from meridian.retrieval.factory import build_retriever

    client = get_qdrant_client()
    item_list = load_retrieval_set(config.retrieval_set_path, config.split)
    model_config_dict = {
        "base": load_config(IndexConfig, config.base_index_config),
        "adapted": load_config(IndexConfig, config.adapted_index_config),
    }
    comparison: dict[str, dict] = {pipeline: {} for pipeline in PIPELINE_MODE_DICT}
    collection_dict: dict[str, dict] = {}
    for model_name, index_config in model_config_dict.items():
        collection, o_built = ensure_collection(
            client, index_config, f"{config.collection_prefix}{model_name}"
        )
        collection_dict[model_name] = {
            "collection": collection,
            "dense_model": index_config.dense.model,
            "built_for_e5": o_built,
        }
        retriever = build_retriever(index_config, client, collection)
        for pipeline, mode in PIPELINE_MODE_DICT.items():
            summary = evaluate_search(mode_search(retriever, mode, config.k), item_list, 3)
            comparison[pipeline][model_name] = {
                key: value for key, value in summary.items() if key != "latency"
            }
        if o_built and config.o_drop_after:
            client.delete_collection(collection)
    delta = {
        pipeline: {
            metric: model_dict["adapted"][metric] - model_dict["base"][metric]
            for metric in ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
        }
        for pipeline, model_dict in comparison.items()
    }
    train_report_path = Path(config.train_report_path)
    train_report = json.loads(train_report_path.read_text()) if train_report_path.exists() else None
    return {
        "dataset": {
            "path": config.retrieval_set_path,
            "split": config.split,
            "n_items": len(item_list),
        },
        "models": collection_dict,
        "comparison": comparison,
        "delta_adapted_minus_base": delta,
        "training": (
            {
                key: train_report[key]
                for key in (
                    "hardware",
                    "device",
                    "wall_clock_s",
                    "train_triplets",
                    "dev_pairs",
                    "batch_size",
                )
            }
            if train_report
            else None
        ),
        "gate_decision": gate_decision(
            config.rebuild_report_dir, model_config_dict["adapted"].dense.model
        ),
    }


def cli_eval(args) -> int:
    """`meridian eval embedder`: run E5 and write its results file."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(E5Config, config_path)
    metrics = run_e5(config)
    write_result(
        config.output,
        metrics,
        config_path,
        notes=(
            "R is held out: no paper referenced by R, A, or O contributed training pairs. "
            "gate_decision is the rebuild gate's verdict on promoting the adapted model; when it "
            "declines, the base model stays live."
        ),
    )
    print(json.dumps(metrics["delta_adapted_minus_base"], indent=2))
    return 0
