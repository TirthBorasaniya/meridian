"""E3: serving behavior on A-fast through the serving runtime.

1. Cold answers with score-band grading: end-to-end latency (p50, p95), tokens, cost, and LLM
   calls per answer, with a fresh LLM response cache so every call reaches the provider.
2. The same questions with score-band grading disabled: LLM calls per answer.
3. Repeated and normalized repeats against the exact answer cache: hit rate and hit latency.
4. The RedisVL semantic cache at `distance_threshold` 0.1: hit rate on same-intent paraphrases
   and wrong-hit rate on different-intent probes.
5. Query-embedding cache hit rate.

Writes `results/e3_serving.json`.
"""

import logging
import time
import uuid

from pydantic import Field

from meridian.common.config import (
    VersionedConfig,
    get_agent_config,
    get_index_config,
    get_models_config,
    get_serving_config,
    load_config,
)
from meridian.common.llm_client import MemoryResponseCache, usage_scope
from meridian.common.results import latency_summary, write_result
from meridian.eval.datasets import load_answer_set, read_jsonl

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/eval/e3.yaml"


class E3Config(VersionedConfig):
    """configs/eval/e3.yaml."""

    answer_set_path: str
    answer_split: str
    cache_probes_path: str
    profile: str
    o_web_fallback: bool
    semantic_distance_threshold: float = Field(gt=0.0)
    output: str


# ============= Runtime construction =============


def make_runtime(config: E3Config, o_score_band: bool, redis_client=None):
    """Build a serving runtime whose agent uses a fresh, empty LLM response cache."""
    from meridian.agent.build import build_agent
    from meridian.common.llm_client import build_llm_client
    from meridian.common.metrics import record_llm_usage
    from meridian.ingest.index import get_qdrant_client
    from meridian.retrieval.factory import build_retriever
    from meridian.serving.runtime import Runtime

    serving_config = get_serving_config(config.profile)
    index_config = get_index_config()
    agent_config = get_agent_config()
    agent_config = agent_config.model_copy(
        update={"grading": agent_config.grading.model_copy(update={"o_score_band": o_score_band})}
    )
    client = get_qdrant_client()
    retriever = build_retriever(index_config, client)
    llm = build_llm_client(cache=MemoryResponseCache(), usage_hook=record_llm_usage)

    def agent_factory():
        return build_agent(
            agent_config=agent_config,
            o_web_fallback=config.o_web_fallback,
            verify_sample_rate=serving_config.verify_sample_rate,
            retriever=retriever,
            llm=llm,
        )

    return Runtime(serving_config, index_config, client, agent_factory, retriever, redis_client)


# ============= Measurements =============


def answer_run(runtime, question_list: list[str]) -> dict:
    """Answer each question in a fresh session and summarize latency, tokens, cost, and calls."""
    models_config = get_models_config()
    latency_list: list[float] = []
    call_list: list[int] = []
    cost_list: list[float] = []
    token_list: list[int] = []
    role_count: dict[str, int] = {}
    status_count: dict[str, int] = {}
    for question in question_list:
        with usage_scope() as meter:
            started = time.perf_counter()
            payload = runtime.ask(question, f"e3-{uuid.uuid4().hex}", None, lambda _event: None)
            latency_list.append(time.perf_counter() - started)
        call_list.append(meter.calls)
        cost_list.append(meter.cost_usd(models_config))
        token_list.append(
            sum(entry["input"] + entry["output"] for entry in meter.tokens_by_model().values())
        )
        for role, count in meter.calls_by_role().items():
            role_count[role] = role_count.get(role, 0) + count
        status_count[payload["status"]] = status_count.get(payload["status"], 0) + 1
    n = len(question_list)
    return {
        "n": n,
        "latency": latency_summary(latency_list),
        "llm_calls_per_answer": sum(call_list) / n if n else None,
        "llm_calls_by_role_per_answer": {role: count / n for role, count in role_count.items()},
        "tokens_per_answer": sum(token_list) / n if n else None,
        "cost_usd_per_answer": sum(cost_list) / n if n else None,
        "status": status_count,
    }


def answer_cache_run(runtime, question_list: list[str]) -> dict:
    """Ask exact and normalized repeats of answered questions; report hits and hit latency."""
    hit = 0
    latency_list: list[float] = []
    variant_list = [question for question in question_list] + [
        f"  {question.lower().rstrip('?')}  " for question in question_list
    ]
    for question in variant_list:
        started = time.perf_counter()
        payload = runtime.ask(question, f"e3-{uuid.uuid4().hex}", None, lambda _event: None)
        if payload.get("cached"):
            hit += 1
            latency_list.append(time.perf_counter() - started)
    return {
        "lookups": len(variant_list),
        "hits": hit,
        "hit_rate": hit / len(variant_list) if variant_list else None,
        "hit_latency": latency_summary(latency_list),
    }


def semantic_cache_run(runtime, probe_list: list[dict], threshold: float, redis_url: str) -> dict:
    """
    Store each probe's cached question, look up its probe, and count hits and wrong hits.

    A wrong hit is a hit for a probe whose intent differs from the cached question, which would
    serve an answer to a different question.
    """
    from meridian.serving.cache import SemanticAnswerCache

    cache = SemanticAnswerCache(
        redis_url,
        runtime.retriever.dense_encoder,
        threshold,
        ttl_seconds=3600,
        name=f"meridian_e3_{uuid.uuid4().hex[:8]}",
    )
    index_version = runtime.index_info().index_version or 0
    same_hits = different_hits = same_total = different_total = 0
    distance_list: list[dict] = []
    try:
        for probe in probe_list:
            cache.clear()
            cache.store(probe["cached_question"], index_version, {"answer": probe["pid"]})
            match = cache.lookup(probe["probe"], index_version)
            o_hit = match is not None
            if probe["same_intent"]:
                same_total += 1
                same_hits += int(o_hit)
            else:
                different_total += 1
                different_hits += int(o_hit)
            distance_list.append(
                {
                    "pid": probe["pid"],
                    "same_intent": probe["same_intent"],
                    "hit": o_hit,
                    "distance": match["semantic_match"]["distance"] if match else None,
                }
            )
    finally:
        cache.clear()
    return {
        "distance_threshold": threshold,
        "same_intent_probes": same_total,
        "same_intent_hit_rate": same_hits / same_total if same_total else None,
        "different_intent_probes": different_total,
        "wrong_hit_rate": different_hits / different_total if different_total else None,
        "probes": distance_list,
    }


def run_e3(config: E3Config) -> dict:
    """Run every E3 measurement."""
    import redis

    from meridian.common.settings import get_settings

    redis_url = get_settings().redis_url
    if not redis_url:
        raise RuntimeError("E3 needs REDIS_URL (Redis Stack) for the semantic cache measurement")
    redis_client = redis.Redis.from_url(redis_url)
    question_list = [
        item["question"] for item in load_answer_set(config.answer_set_path, config.answer_split)
    ]

    band_runtime = make_runtime(config, o_score_band=True)
    score_band = answer_run(band_runtime, question_list)
    answer_cache = answer_cache_run(band_runtime, question_list)
    embedding_cache = {
        "hits": band_runtime.embedding_cache.hits,
        "misses": band_runtime.embedding_cache.misses,
        "hit_rate": band_runtime.embedding_cache.hits
        / max(1, band_runtime.embedding_cache.hits + band_runtime.embedding_cache.misses),
    }
    no_band_runtime = make_runtime(config, o_score_band=False)
    no_score_band = answer_run(no_band_runtime, question_list)
    semantic = semantic_cache_run(
        band_runtime,
        read_jsonl(config.cache_probes_path),
        config.semantic_distance_threshold,
        redis_url,
    )
    redis_client.close()
    models_config = get_models_config()
    return {
        "questions": {
            "path": config.answer_set_path,
            "split": config.answer_split,
            "n": len(question_list),
        },
        "index_version": band_runtime.index_info().index_version,
        "models": {role: models_config.role(role).model for role in models_config.roles},
        "prices": {
            "usd_per_million_tokens": {
                model: price.model_dump()
                for model, price in models_config.prices_usd_per_million_tokens.items()
            },
            "checked_on": str(models_config.prices_checked_on),
        },
        "score_band_grading": score_band,
        "llm_grading_only": no_score_band,
        "answer_cache": answer_cache,
        "embedding_cache": embedding_cache,
        "semantic_cache": semantic,
    }


def cli_eval(args) -> int:
    """`meridian eval serving`: run E3 and write its results file."""
    import json

    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(E3Config, config_path)
    metrics = run_e3(config)
    write_result(
        config.output,
        metrics,
        config_path,
        notes=(
            "Every answer run uses a fresh LLM response cache, so calls, tokens, and latency are "
            "uncached. Cost converts measured tokens with the configured prices; see "
            "prices.checked_on. Semantic cache probes are hand-written question pairs."
        ),
    )
    print(json.dumps({key: metrics[key] for key in ("answer_cache", "embedding_cache")}, indent=2))
    return 0
