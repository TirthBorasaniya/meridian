"""Process-wide serving state: Qdrant, Redis or in-process caches, the retriever, the agent, the
rate limiter, and live index information. `ask` wraps the agent with the answer caches and the
metrics.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from meridian.agent.citations import strip_citations
from meridian.common import metrics
from meridian.common.config import (
    IndexConfig,
    ServingConfig,
    get_corpus_config,
    get_index_config,
    get_serving_config,
)
from meridian.common.settings import get_settings
from meridian.ingest import index as index_module
from meridian.serving.cache import AnswerCache, EmbeddingCache, MemoryStore, RedisStore
from meridian.serving.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

INDEX_INFO_TTL_S = 60.0
CACHEABLE_STATUS_SET = {"answered", "insufficient", "out_of_scope"}


@dataclass
class IndexInfo:
    """What `/v1/index` and the UI footer report."""

    alias: str
    collection: str | None
    index_version: int | None
    papers: int
    points: int
    last_ingest: str | None
    snapshot: str | None


class Runtime:
    """Serving dependencies, built once per process."""

    def __init__(
        self,
        serving_config: ServingConfig,
        index_config: IndexConfig,
        client,
        agent_factory: Callable[[], object],
        retriever,
        redis_client=None,
        semantic_cache=None,
    ) -> None:
        self.serving_config = serving_config
        self.index_config = index_config
        self.client = client
        self._agent_factory = agent_factory
        self._agent = None
        self.retriever = retriever
        store = RedisStore(redis_client) if redis_client is not None else MemoryStore()
        self.embedding_cache = EmbeddingCache(store, serving_config.cache.embedding_ttl_seconds)
        self.answer_cache = AnswerCache(store, serving_config.cache.answer_ttl_seconds)
        self.semantic_cache = semantic_cache
        self.rate_limiter = RateLimiter(serving_config.rate_limit, redis_client)
        self._index_info: IndexInfo | None = None
        self._index_info_at = 0.0
        encoder = getattr(retriever, "dense_encoder", None)
        if encoder is not None and hasattr(encoder, "cache"):
            encoder.cache = self.embedding_cache

    @property
    def agent(self):
        """The agent, built on first use."""
        if self._agent is None:
            self._agent = self._agent_factory()
        return self._agent

    # ---------- index ----------

    def index_info(self, o_refresh: bool = False) -> IndexInfo:
        """Return live index information, refreshed at most once a minute."""
        now = time.monotonic()
        if (
            self._index_info is not None
            and not o_refresh
            and now - self._index_info_at < INDEX_INFO_TTL_S
        ):
            return self._index_info
        alias = self.index_config.alias
        collection = index_module.alias_target(self.client, alias)
        papers = points = 0
        if collection is not None:
            points = index_module.count_points(self.client, collection)
            papers = index_module.count_papers(self.client, collection)
        metrics.INDEX_POINTS.set(points)
        info_path = Path(self.serving_config.index_info_path)
        info_dict = json.loads(info_path.read_text()) if info_path.exists() else {}
        self._index_info = IndexInfo(
            alias=alias,
            collection=collection,
            index_version=(
                index_module.parse_version(collection, self.index_config.collection_prefix)
                if collection
                else None
            ),
            papers=papers,
            points=points,
            last_ingest=info_dict.get("last_ingest") or self._manifest_last_ingest(),
            snapshot=info_dict.get("snapshot"),
        )
        self._index_info_at = now
        return self._index_info

    @staticmethod
    def _manifest_last_ingest() -> str | None:
        manifest_path = Path(get_corpus_config().paths.manifest)
        if not manifest_path.exists():
            return None
        from meridian.ingest.manifest import Manifest

        manifest = Manifest(manifest_path)
        try:
            accepted_list = [
                batch for batch in manifest.batches() if batch["decision"] == "accepted"
            ]
        finally:
            manifest.close()
        return accepted_list[-1]["finished_at"] if accepted_list else None

    # ---------- ask ----------

    def ask(
        self,
        question: str,
        session_id: str,
        paper_filter: list[str] | None,
        emit: Callable[[dict], None],
    ) -> dict:
        """
        Answer a question with caching, streaming, and metrics.

        Parameters
        ----------
        question : str
            The user's question.
        session_id : str
            Conversation key.
        paper_filter : list of str or None
            Pinned papers.
        emit : callable
            Receives `token` and `reset` events while the answer is generated.

        Returns
        -------
        payload : dict
            Answer, status, citations, session id, index version, cache flag, and timings.
        """
        started = time.perf_counter()
        index_version = self.index_info().index_version or 0
        o_session_independent = self.agent.history_length(session_id) == 0 and not paper_filter
        cache_config = self.serving_config.cache
        if o_session_independent and cache_config.o_answer_cache:
            cached = self.answer_cache.get(question, True, index_version)
            if cached is None and self.semantic_cache is not None and cache_config.o_semantic_cache:
                cached = self.semantic_cache.lookup(question, index_version)
            if cached is not None:
                emit({"type": "token", "text": cached["answer"]})
                self.agent.record_turn(session_id, question, strip_citations(cached["answer"]))
                metrics.REQUESTS_TOTAL.labels(route="cache").inc()
                return {
                    **cached,
                    "session_id": session_id,
                    "cached": True,
                    "latency_s": time.perf_counter() - started,
                }

        result = self.agent.ask(question, session_id, paper_filter, on_event=emit)
        metrics.REQUESTS_TOTAL.labels(route=result.route or "unknown").inc()
        for stage, seconds in result.timings.items():
            metrics.STAGE_LATENCY_SECONDS.labels(stage=stage).observe(seconds)
        metrics.STAGE_LATENCY_SECONDS.labels(stage="end_to_end").observe(result.latency_s)
        if result.route == "answer":
            metrics.RETRIEVAL_SUPPORTING_CHUNKS.observe(len(result.graded))
        if result.fallback:
            metrics.FALLBACK_TOTAL.labels(kind=result.fallback).inc()
        if result.retries:
            metrics.FALLBACK_TOTAL.labels(kind="rewrite_retry").inc(result.retries)
        payload = {
            "answer": result.answer,
            "status": result.status,
            "citations": result.citations,
            "route": result.route,
            "fallback": result.fallback,
            "index_version": index_version,
        }
        if (
            o_session_independent
            and cache_config.o_answer_cache
            and result.status in CACHEABLE_STATUS_SET
            and result.fallback != "web"
        ):
            self.answer_cache.put(question, True, index_version, payload)
            if self.semantic_cache is not None and cache_config.o_semantic_cache:
                self.semantic_cache.store(question, index_version, payload)
        return {
            **payload,
            "session_id": session_id,
            "cached": False,
            "latency_s": result.latency_s,
            "timings": result.timings,
            "llm_calls": result.usage.calls,
        }

    # ---------- search ----------

    def search(self, query: str, k: int, mode: str, paper_filter: list[str] | None) -> dict:
        """Run retrieval only and return chunks with scores and stage timings."""
        from meridian.agent.citations import excerpt

        result = self.retriever.search(query, mode=mode, k=k, paper_filter=paper_filter)
        for stage, seconds in result.timing_dict.items():
            metrics.STAGE_LATENCY_SECONDS.labels(stage=f"search_{stage}").observe(seconds)
        metrics.REQUESTS_TOTAL.labels(route="search").inc()
        chunk_list = []
        for chunk in result.chunk_list:
            payload = chunk.payload
            item = {
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "rerank_score": chunk.rerank_score,
                "arxiv_id": payload.get("arxiv_id"),
                "version": payload.get("version"),
                "title": payload.get("title"),
                "section_title": payload.get("section_title"),
                "url": f"https://arxiv.org/abs/{payload.get('arxiv_id')}v{payload.get('version')}",
                "excerpt": excerpt(payload.get("text", "")),
            }
            if self.serving_config.o_expose_chunk_text:
                item["text"] = payload.get("text", "")
            chunk_list.append(item)
        return {"query": query, "mode": mode, "chunks": chunk_list, "timings": result.timing_dict}


def index_info_dict(info: IndexInfo) -> dict:
    """Serialize index information."""
    return asdict(info)


def build_runtime(profile: str | None = None) -> Runtime:
    """Build the production runtime from settings and the serving profile."""
    from meridian.agent.build import build_agent
    from meridian.common.llm_client import RedisResponseCache, build_llm_client
    from meridian.retrieval.factory import build_retriever

    settings = get_settings()
    serving_config = get_serving_config(profile or settings.serving_profile)
    index_config = get_index_config()
    client = index_module.get_qdrant_client()
    redis_client = None
    if settings.redis_url:
        import redis

        redis_client = redis.Redis.from_url(settings.redis_url)
    retriever = build_retriever(index_config, client)
    semantic_cache = None
    if serving_config.cache.o_semantic_cache and settings.redis_url:
        from meridian.serving.cache import SemanticAnswerCache

        semantic_cache = SemanticAnswerCache(
            settings.redis_url,
            retriever.dense_encoder,
            serving_config.cache.semantic_distance_threshold,
            serving_config.cache.semantic_ttl_seconds,
        )

    def agent_factory():
        llm_cache = RedisResponseCache(redis_client) if redis_client is not None else None
        llm = build_llm_client(cache=llm_cache, usage_hook=metrics.record_llm_usage)
        return build_agent(
            o_web_fallback=serving_config.o_web_fallback,
            verify_sample_rate=serving_config.verify_sample_rate,
            retriever=retriever,
            llm=llm,
        )

    return Runtime(
        serving_config,
        index_config,
        client,
        agent_factory,
        retriever,
        redis_client=redis_client,
        semantic_cache=semantic_cache,
    )
