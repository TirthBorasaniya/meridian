"""Prometheus metrics with the `meridian_` prefix (SPEC S5)."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from meridian.common.llm_client import LLMResponse

REGISTRY = CollectorRegistry(auto_describe=True)
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0)

REQUESTS_TOTAL = Counter(
    "meridian_requests_total", "Requests by route", ["route"], registry=REGISTRY
)
STAGE_LATENCY_SECONDS = Histogram(
    "meridian_stage_latency_seconds",
    "Latency of each pipeline stage",
    ["stage"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
RETRIEVAL_SUPPORTING_CHUNKS = Histogram(
    "meridian_retrieval_supporting_chunks",
    "Chunks that survived grading per answer",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10),
    registry=REGISTRY,
)
FALLBACK_TOTAL = Counter(
    "meridian_fallback_total", "Fallbacks by kind", ["kind"], registry=REGISTRY
)
CACHE_HITS_TOTAL = Counter(
    "meridian_cache_hits_total", "Cache hits by cache", ["cache"], registry=REGISTRY
)
CACHE_MISSES_TOTAL = Counter(
    "meridian_cache_misses_total", "Cache misses by cache", ["cache"], registry=REGISTRY
)
LLM_TOKENS_TOTAL = Counter(
    "meridian_llm_tokens_total",
    "LLM tokens by model and direction",
    ["model", "direction"],
    registry=REGISTRY,
)
INGEST_BATCHES_TOTAL = Counter(
    "meridian_ingest_batches_total", "Ingest batches by decision", ["decision"], registry=REGISTRY
)
INDEX_POINTS = Gauge("meridian_index_points", "Points in the live collection", registry=REGISTRY)


def record_llm_usage(response: LLMResponse) -> None:
    """LLM client usage hook: count tokens of calls that reached the provider."""
    if response.o_cached:
        return
    LLM_TOKENS_TOTAL.labels(model=response.model, direction="input").inc(response.input_tokens)
    LLM_TOKENS_TOTAL.labels(model=response.model, direction="output").inc(response.output_tokens)


def record_cache(cache_name: str, o_hit: bool) -> None:
    """Count one cache lookup."""
    (CACHE_HITS_TOTAL if o_hit else CACHE_MISSES_TOTAL).labels(cache=cache_name).inc()
