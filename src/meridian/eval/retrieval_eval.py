"""E1: retrieval quality (recall@5, recall@10, MRR@10, nDCG@10) and p50 and p95 latency per stage
for dense only, BM25 only, hybrid RRF, hybrid RRF plus rerank, and the previous in-process BM25
path, plus prefetch-limit and chunk-size sensitivity. Writes `results/e1_retrieval.json`.

`evaluate_search` is the shared core used by the daily gate and by E5.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from meridian.common.config import (
    ChunkerConfig,
    VersionedConfig,
    get_corpus_config,
    get_index_config,
    load_config,
)
from meridian.common.results import latency_summary, write_result
from meridian.eval.datasets import RetrievalItem, count_invalid, load_retrieval_set
from meridian.eval.metrics import mean_metrics, query_metrics
from meridian.retrieval.hybrid import STAGE_LIST, RetrievalResult

logger = logging.getLogger(__name__)

# ============= Config =============

DEFAULT_CONFIG_PATH = "configs/eval/e1.yaml"


class PrefetchSensitivity(BaseModel):
    """Prefetch limits evaluated in one mode."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    limits: list[int] = Field(min_length=1)


class ChunkSizeSensitivity(BaseModel):
    """Chunk sizes evaluated through temporary collections."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    chunk_tokens: list[int] = Field(min_length=1)
    overlap_fraction: float = Field(ge=0.0, lt=1.0)
    collection_prefix: str
    o_drop_after: bool


class E1Config(VersionedConfig):
    """configs/eval/e1.yaml."""

    retrieval_set_path: str
    split: str
    k: int = Field(ge=10)
    warmup_queries: int = Field(ge=0)
    configurations: list[str]
    prefetch_sensitivity: PrefetchSensitivity
    chunk_size_sensitivity: ChunkSizeSensitivity
    output: str


# ============= Core =============

SearchFunction = Callable[[str], RetrievalResult]


def evaluate_search(
    search: SearchFunction,
    item_list: list[RetrievalItem],
    warmup_queries: int = 0,
    o_keep_per_query: bool = False,
) -> dict:
    """
    Run every query through a search function and aggregate metrics and latency.

    Parameters
    ----------
    search : callable
        Maps a question to a `RetrievalResult` with at least 10 chunks when available.
    item_list : list of RetrievalItem
        Labeled queries.
    warmup_queries : int
        Leading queries run once and excluded from latency, to absorb model warm-up.
    o_keep_per_query : bool
        Include per-query metrics in the summary.

    Returns
    -------
    summary : dict
        Mean metrics, `n_queries`, `latency` per stage (count, p50_s, p95_s, mean_s), and
        optionally `per_query`.
    """
    for item in item_list[:warmup_queries]:
        search(item.question)
    per_query_list: list[dict] = []
    stage_latency_dict: dict[str, list[float]] = {stage: [] for stage in STAGE_LIST}
    for item in item_list:
        result = search(item.question)
        metrics = query_metrics([chunk.payload for chunk in result.chunk_list], item.span_list)
        per_query_list.append({"qid": item.qid, **metrics})
        for stage in STAGE_LIST:
            stage_latency_dict[stage].append(result.timing_dict.get(stage, 0.0))
    summary = {
        **mean_metrics([{k: v for k, v in row.items() if k != "qid"} for row in per_query_list]),
        "n_queries": len(item_list),
        "latency": {stage: latency_summary(values) for stage, values in stage_latency_dict.items()},
    }
    if o_keep_per_query:
        summary["per_query"] = per_query_list
    return summary


def mode_search(retriever, mode: str, k: int, prefetch_limit: int | None = None) -> SearchFunction:
    """Return a search function bound to one retriever mode."""

    def search(query: str) -> RetrievalResult:
        return retriever.search(query, mode=mode, k=k, prefetch_limit=prefetch_limit)

    return search


def legacy_search(legacy, k: int) -> SearchFunction:
    """Return a search function over the legacy retriever."""

    def search(query: str) -> RetrievalResult:
        return legacy.search(query, k)

    return search


# ============= Sensitivity collections =============


def scaled_chunker(
    base: ChunkerConfig, chunk_tokens: int, overlap_fraction: float
) -> ChunkerConfig:
    """Return a chunker config with a new window and proportional overlap."""
    return base.model_copy(
        update={
            "chunk_tokens": chunk_tokens,
            "overlap_tokens": int(round(chunk_tokens * overlap_fraction)),
        }
    )


def build_sensitivity_collection(client, manifest, chunker: ChunkerConfig, name: str) -> dict:
    """Build a temporary collection with another chunker and return its build summary."""
    from meridian.ingest.build import build_collection
    from meridian.ingest.index import live_version

    index_config = get_index_config()
    variant = index_config.model_copy(update={"chunker": chunker})
    if client.collection_exists(name):
        client.delete_collection(name)
    report = build_collection(
        client,
        manifest,
        variant,
        get_corpus_config(),
        collection=name,
        index_version=live_version(client, index_config) or 1,
    )
    return {"collection": name, "papers": report.papers_indexed, "chunks": report.chunks_indexed}


# ============= Runner =============


def run_e1(config: E1Config) -> dict:
    """Run every E1 configuration and sensitivity sweep and return the metrics block."""
    from meridian.ingest import index as index_module
    from meridian.ingest.embed import get_dense_encoder
    from meridian.ingest.manifest import Manifest
    from meridian.retrieval.factory import build_retriever
    from meridian.retrieval.legacy import LegacyRetriever
    from meridian.retrieval.rerank import get_reranker

    index_config = get_index_config()
    client = index_module.get_qdrant_client()
    live = index_module.alias_target(client, index_config.alias)
    if live is None:
        raise RuntimeError(f"alias {index_config.alias} does not exist")
    item_list = load_retrieval_set(config.retrieval_set_path, config.split)
    retriever = build_retriever(index_config, client, live)

    configuration_dict: dict[str, dict] = {}
    for name in config.configurations:
        logger.info("e1 configuration", extra={"configuration": name})
        if name == "legacy":
            legacy = LegacyRetriever(
                client,
                live,
                get_dense_encoder(index_config.dense),
                get_reranker(index_config.reranker),
            )
            search = legacy_search(legacy, config.k)
        else:
            search = mode_search(retriever, name, config.k)
        configuration_dict[name] = evaluate_search(
            search, item_list, config.warmup_queries, o_keep_per_query=True
        )

    prefetch_dict: dict[str, dict] = {}
    for limit in config.prefetch_sensitivity.limits:
        prefetch_dict[str(limit)] = evaluate_search(
            mode_search(retriever, config.prefetch_sensitivity.mode, config.k, limit),
            item_list,
            config.warmup_queries,
        )

    chunk_dict: dict[str, dict] = {}
    sensitivity = config.chunk_size_sensitivity
    manifest = Manifest(get_corpus_config().paths.manifest)
    for chunk_tokens in sensitivity.chunk_tokens:
        if chunk_tokens == index_config.chunker.chunk_tokens:
            collection = live
            build_summary = {"collection": live, "reused_live": True}
        else:
            collection = f"{sensitivity.collection_prefix}{chunk_tokens}"
            build_summary = build_sensitivity_collection(
                client,
                manifest,
                scaled_chunker(index_config.chunker, chunk_tokens, sensitivity.overlap_fraction),
                collection,
            )
        variant_retriever = build_retriever(index_config, client, collection)
        chunk_dict[str(chunk_tokens)] = {
            **evaluate_search(
                mode_search(variant_retriever, sensitivity.mode, config.k),
                item_list,
                config.warmup_queries,
            ),
            "build": build_summary,
        }
        if sensitivity.o_drop_after and collection != live:
            client.delete_collection(collection)

    return {
        "dataset": {
            "path": config.retrieval_set_path,
            "split": config.split,
            "n_items": len(item_list),
            "n_invalid_excluded": count_invalid(config.retrieval_set_path),
        },
        "index": {
            "collection": live,
            "points": index_module.count_points(client, live),
            "papers": index_module.count_papers(client, live),
            "metadata": index_module.collection_metadata(client, live),
        },
        "k": config.k,
        "configurations": configuration_dict,
        "prefetch_sensitivity": {
            "mode": config.prefetch_sensitivity.mode,
            "by_limit": prefetch_dict,
        },
        "chunk_size_sensitivity": {"mode": sensitivity.mode, "by_chunk_tokens": chunk_dict},
    }


def cli_eval(args) -> int:
    """`meridian eval retrieval`: run E1 and write its results file."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(E1Config, config_path)
    metrics = run_e1(config)
    output_path = write_result(
        config.output,
        metrics,
        config_path,
        notes=(
            "Relevance: same paper version and at least 50% coverage of a labeled span. "
            "Latency excludes warm-up queries. Legacy is dense top-20 plus in-process rank_bm25 "
            "top-20, Python RRF k=60, reranked from the fused top-20."
        ),
    )
    print(Path(output_path).read_text()[:2000])
    return 0
