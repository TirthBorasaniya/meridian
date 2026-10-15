"""Eval-gated rebuilds (SPEC S3.7): build `meridian_v{N+1}`, run the full evaluation on the
candidate and on the live version, and move the alias atomically only when the candidate is
not worse on recall@5 and nDCG@10 by more than `max_regression` and not worse on faithfulness
by more than 0.02. One previous version is kept for rollback.
"""

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient

from meridian.common.config import (
    CorpusConfig,
    GateConfig,
    IndexConfig,
    get_corpus_config,
    get_gate_config,
    get_index_config,
    load_config,
)
from meridian.eval.datasets import SPLIT_FULL, load_retrieval_set
from meridian.eval.retrieval_eval import SearchFunction, evaluate_search
from meridian.ingest import index as index_module
from meridian.ingest.build import build_collection, record_build_in_manifest
from meridian.ingest.gate import decide_promotion, run_fast_gate, save_baseline
from meridian.ingest.manifest import Manifest

logger = logging.getLogger(__name__)

# ============= Types =============

# maps (collection, index config of that collection) to a full-pipeline search function
SearchFactory = Callable[[str, IndexConfig], SearchFunction]
# maps (collection, index config of that collection) to A-full RAGAS faithfulness
AnswerEvaluator = Callable[[str, IndexConfig], float]

REBUILD_REPORT_DIR = Path("data/rebuilds")


def default_search_factory(client: QdrantClient) -> SearchFactory:
    """Return a factory of hybrid-plus-rerank search functions over one collection."""

    def factory(collection: str, index_config: IndexConfig) -> SearchFunction:
        from meridian.retrieval.factory import build_retriever

        retriever = build_retriever(index_config, client, collection)

        def search(query: str):
            return retriever.search(query, mode="hybrid_rerank", k=10)

        return search

    return factory


def default_answer_evaluator(collection: str, index_config: IndexConfig) -> float:
    """Return A-full faithfulness of the agent answering from one collection."""
    from meridian.eval.answer_eval import faithfulness_on_collection

    return faithfulness_on_collection(collection, index_config)


def config_for_collection(base_config: IndexConfig, metadata: dict) -> IndexConfig:
    """
    Return the index config a collection was built with, as far as querying depends on it.

    The dense model is read from the collection metadata, so the live version is queried with
    its own model even when the candidate uses another one.
    """
    dense_model = metadata.get("dense_model")
    if not dense_model or dense_model == base_config.dense.model:
        return base_config
    return base_config.model_copy(
        update={"dense": base_config.dense.model_copy(update={"model": dense_model})}
    )


# ============= Report =============


@dataclass
class RebuildReport:
    """Outcome of one rebuild."""

    candidate: str
    live_before: str | None
    build: dict
    retrieval: dict
    faithfulness: dict
    o_promoted: bool
    reason_list: list[str]
    live_after: str | None
    pruned_list: list[str] = field(default_factory=list)
    baseline: dict | None = None
    dense_model: str = ""

    def to_dict(self) -> dict:
        """Serialize."""
        return asdict(self)


# ============= Rebuild =============


def run_rebuild(
    client: QdrantClient,
    manifest: Manifest,
    candidate_config: IndexConfig,
    corpus_config: CorpusConfig,
    gate_config: GateConfig,
    search_factory: SearchFactory,
    answer_evaluator: AnswerEvaluator | None,
    o_dry_run: bool = False,
    indexer_factory=None,
    baseline_path: str | None = None,
    retrieval_set_path: str | None = None,
) -> RebuildReport:
    """
    Build a new version, compare it with the live version, and promote it when the gate passes.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client.
    manifest : Manifest
        Corpus papers to index.
    candidate_config : IndexConfig
        Index configuration of the new version (chunker, models).
    corpus_config : CorpusConfig
        Corpus paths.
    gate_config : GateConfig
        Tolerances and dataset paths.
    search_factory : callable
        Builds full-pipeline search functions for R-full.
    answer_evaluator : callable or None
        Computes A-full faithfulness; None skips it, which declines promotion.
    o_dry_run : bool
        Evaluate but never move the alias.
    indexer_factory : callable or None
        Passed to `build_collection`.
    baseline_path, retrieval_set_path : str or None
        Overrides of the gate config paths.

    Returns
    -------
    report : RebuildReport
        Also written to `data/rebuilds/{candidate}.json` by the CLI.
    """
    alias = candidate_config.alias
    live = index_module.alias_target(client, alias)
    build_batch_id = f"build-{uuid.uuid4().hex[:12]}"
    build_report = build_collection(
        client,
        manifest,
        candidate_config,
        corpus_config,
        build_batch_id=build_batch_id,
        indexer_factory=indexer_factory,
    )
    candidate = build_report.collection
    set_path = retrieval_set_path or gate_config.retrieval_set_path
    item_list = load_retrieval_set(set_path, SPLIT_FULL)

    candidate_metrics = evaluate_search(search_factory(candidate, candidate_config), item_list)
    live_config = (
        config_for_collection(candidate_config, index_module.collection_metadata(client, live))
        if live
        else None
    )
    live_metrics = (
        evaluate_search(search_factory(live, live_config), item_list)
        if live and live_config
        else None
    )
    candidate_faithfulness = (
        answer_evaluator(candidate, candidate_config) if answer_evaluator else None
    )
    live_faithfulness = (
        answer_evaluator(live, live_config) if answer_evaluator and live and live_config else None
    )
    o_promoted, reason_list = decide_promotion(
        candidate_metrics, live_metrics, candidate_faithfulness, live_faithfulness, gate_config
    )

    def strip(metrics: dict | None) -> dict | None:
        return (
            {key: value for key, value in metrics.items() if key not in {"latency", "per_query"}}
            if metrics
            else None
        )

    report = RebuildReport(
        candidate=candidate,
        live_before=live,
        build={
            "papers": build_report.papers_indexed,
            "chunks": build_report.chunks_indexed,
            "seconds": build_report.seconds,
            "failures": build_report.failure_dict,
        },
        retrieval={"candidate": strip(candidate_metrics), "live": strip(live_metrics)},
        faithfulness={"candidate": candidate_faithfulness, "live": live_faithfulness},
        o_promoted=o_promoted and not o_dry_run,
        reason_list=reason_list + (["dry run"] if o_dry_run else []),
        live_after=live,
        dense_model=candidate_config.dense.model,
    )
    if not report.o_promoted:
        logger.warning(
            "rebuild declined", extra={"candidate": candidate, "reasons": report.reason_list}
        )
        return report

    index_module.switch_alias(client, alias, candidate)
    record_build_in_manifest(manifest, build_report, build_batch_id)
    report.live_after = candidate
    report.pruned_list = index_module.prune_versions(client, candidate_config)
    # the fast-gate baseline must describe the version that is now live
    fast_result = run_fast_gate(
        search_factory(candidate, candidate_config),
        gate_config,
        candidate,
        baseline_path,
        set_path,
        o_ignore_baseline=True,
    )
    report.baseline = save_baseline(
        baseline_path or gate_config.baseline_path, fast_result, set_path
    )
    logger.info("rebuild promoted", extra={"candidate": candidate, "previous": live})
    return report


# ============= Rollback and switch =============


def rollback(client: QdrantClient, index_config: IndexConfig) -> tuple[str | None, str]:
    """
    Point the alias at the newest kept version older than the live one.

    Returns
    -------
    switch : tuple of (str or None, str)
        (previous live collection, new live collection).

    Raises
    ------
    RuntimeError
        If there is no older version to roll back to.
    """
    current = index_module.live_version(client, index_config)
    older_list = [
        version
        for version in index_module.list_versions(client, index_config.collection_prefix)
        if current is None or version < current
    ]
    if not older_list:
        raise RuntimeError("no previous version is kept; nothing to roll back to")
    target = index_module.collection_name(index_config.collection_prefix, older_list[-1])
    previous = index_module.switch_alias(client, index_config.alias, target)
    return previous, target


# ============= CLI =============


def cli_rebuild(args) -> int:
    """`meridian rebuild`: exit 0 when promoted, 2 when the gate declined."""
    candidate_config = (
        load_config(IndexConfig, args.index_config) if args.index_config else get_index_config()
    )
    client = index_module.get_qdrant_client()
    corpus_config = get_corpus_config()
    report = run_rebuild(
        client,
        Manifest(corpus_config.paths.manifest),
        candidate_config,
        corpus_config,
        get_gate_config(),
        default_search_factory(client),
        None if args.o_skip_answer_eval else default_answer_evaluator,
        o_dry_run=args.o_dry_run,
    )
    REBUILD_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REBUILD_REPORT_DIR / f"{report.candidate}.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n"
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.o_promoted else 2


def cli_switch(args) -> int:
    """`meridian index switch <collection>`: manual atomic alias switch."""
    index_config = get_index_config()
    previous = index_module.switch_alias(
        index_module.get_qdrant_client(), index_config.alias, args.collection
    )
    print(json.dumps({"alias": index_config.alias, "previous": previous, "live": args.collection}))
    return 0


def cli_rollback(args) -> int:
    """`meridian rollback`: point the alias at the previous kept version."""
    previous, target = rollback(index_module.get_qdrant_client(), get_index_config())
    print(json.dumps({"previous": previous, "live": target}))
    return 0
