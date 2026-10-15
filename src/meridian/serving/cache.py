"""Serving caches: query embeddings keyed by model and text hash; answers keyed by normalized
query, the session-independence flag, and the index version; and an optional RedisVL semantic
answer cache filtered on index version. Keying answers on the index version means a new index
invalidates every cached answer, so a cached answer never cites a stale index.
"""

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ============= Constants =============

MEMORY_STORE_MAX_ITEMS = 50_000
TRAILING_PUNCTUATION = re.compile(r"[\s?.!]+$")


# ============= Stores =============


class KeyValueStore(Protocol):
    """String key-value store with per-key TTL."""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...


class MemoryStore:
    """Bounded in-process store with TTL, used when no Redis is configured."""

    def __init__(self, max_items: int = MEMORY_STORE_MAX_ITEMS, clock=time.monotonic) -> None:
        self._data: OrderedDict[str, tuple[str, float | None]] = OrderedDict()
        self._max_items = max_items
        self._clock = clock
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        """Return a live value or None."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and self._clock() >= expires_at:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Store a value, evicting the least recently used entry when full."""
        with self._lock:
            expires_at = self._clock() + ttl_seconds if ttl_seconds else None
            self._data[key] = (value, expires_at)
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)


class RedisStore:
    """Redis-backed store."""

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def get(self, key: str) -> str | None:
        """Return a value or None."""
        raw = self._redis.get(key)
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Store a value."""
        self._redis.set(key, value, ex=ttl_seconds)


# ============= Embedding cache =============


class EmbeddingCache:
    """Query embedding cache; implements the encoder's `VectorCache` protocol."""

    def __init__(self, store: KeyValueStore, ttl_seconds: int) -> None:
        self._store = store
        self._ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get_vector(self, key: str) -> list[float] | None:
        """Return a cached vector; keys come from `embedding_cache_key`."""
        from meridian.common.metrics import record_cache

        raw = self._store.get(key)
        record_cache("embedding", raw is not None)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set_vector(self, key: str, vector: list[float]) -> None:
        """Store a vector."""
        self._store.set(key, json.dumps(vector), self._ttl_seconds)


# ============= Answer cache =============


def normalize_question(question: str) -> str:
    """Lowercase, collapse whitespace, and drop trailing punctuation."""
    return TRAILING_PUNCTUATION.sub("", " ".join(question.lower().split()))


def answer_cache_key(question: str, o_session_independent: bool, index_version: int | str) -> str:
    """
    Return the answer cache key.

    Parameters
    ----------
    question : str
        The user's question.
    o_session_independent : bool
        True when the answer does not depend on earlier turns of a session.
    index_version : int or str
        Live index revision (version plus last ingest); any change yields new keys, so an
        answer is never served from an index it was not generated on.

    Returns
    -------
    key : str
        `ans:v{index_version}:si{0|1}:{sha256 of the normalized question}`.
    """
    digest = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
    return f"ans:v{index_version}:si{int(o_session_independent)}:{digest}"


class AnswerCache:
    """Exact-match answer cache over a key-value store."""

    def __init__(self, store: KeyValueStore, ttl_seconds: int) -> None:
        self._store = store
        self._ttl_seconds = ttl_seconds

    def get(
        self, question: str, o_session_independent: bool, index_version: int | str
    ) -> dict | None:
        """Return a cached answer payload."""
        from meridian.common.metrics import record_cache

        raw = self._store.get(answer_cache_key(question, o_session_independent, index_version))
        record_cache("answer", raw is not None)
        return json.loads(raw) if raw else None

    def put(
        self, question: str, o_session_independent: bool, index_version: int | str, payload: dict
    ) -> None:
        """Store an answer payload."""
        self._store.set(
            answer_cache_key(question, o_session_independent, index_version),
            json.dumps(payload),
            self._ttl_seconds,
        )


# ============= Semantic cache =============


class SemanticAnswerCache:
    """RedisVL `SemanticCache` over query embeddings, filtered on index version; off by default."""

    def __init__(
        self,
        redis_url: str,
        dense_encoder,
        distance_threshold: float,
        ttl_seconds: int,
        name: str = "meridian_semantic",
    ) -> None:
        from redisvl.extensions.cache.llm import SemanticCache
        from redisvl.utils.vectorize import CustomTextVectorizer

        self._encoder = dense_encoder
        self._cache = SemanticCache(
            name=name,
            distance_threshold=distance_threshold,
            ttl=ttl_seconds,
            vectorizer=CustomTextVectorizer(embed=dense_encoder.encode_query),
            filterable_fields=[{"name": "index_version", "type": "tag"}],
            redis_url=redis_url,
        )

    def lookup(self, question: str, index_version: int | str) -> dict[str, Any] | None:
        """Return the closest cached answer within the threshold for this index version."""
        from redisvl.query.filter import Tag

        from meridian.common.metrics import record_cache

        hit_list = self._cache.check(
            vector=self._encoder.encode_query(question),
            num_results=1,
            filter_expression=Tag("index_version") == str(index_version),
        )
        record_cache("semantic", bool(hit_list))
        if not hit_list:
            return None
        hit = hit_list[0]
        payload = json.loads(hit["response"])
        payload["semantic_match"] = {
            "cached_question": hit.get("prompt", ""),
            "distance": float(hit.get("vector_distance", 0.0)),
        }
        return payload

    def store(self, question: str, index_version: int | str, payload: dict) -> None:
        """Store an answer under this index version."""
        self._cache.store(
            prompt=question,
            response=json.dumps(payload),
            vector=self._encoder.encode_query(question),
            filters={"index_version": str(index_version)},
        )

    def clear(self) -> None:
        """Remove every cached entry."""
        self._cache.clear()
