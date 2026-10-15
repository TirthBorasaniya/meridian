"""Unit tests for serving: cache keys carry the index version, TTL stores, rate limits, SSE
framing, and the API endpoints with an in-memory collection and a scripted agent."""

import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from meridian.common.config import RateLimitConfig, get_index_config, get_serving_config
from meridian.ingest import index as index_module
from meridian.serving.app import create_app
from meridian.serving.cache import AnswerCache, MemoryStore, answer_cache_key, normalize_question
from meridian.serving.rate_limit import RateLimiter
from meridian.serving.runtime import Runtime
from meridian.serving.sse import sse_event
from tests.agent_fakes import FakeRetriever, Script, make_agent, make_chunk
from tests.fakes import FakeDenseEncoder, FakeSparseEncoder

pytestmark = pytest.mark.unit

CHUNKS = [make_chunk(1, 0.95), make_chunk(2, 0.9)]


def test_answer_cache_key_includes_index_version_and_session_flag():
    key = answer_cache_key("What is CoT?", True, 3)
    assert key.startswith("ans:v3:si1:")
    assert key == answer_cache_key("  what is cot ", True, 3)
    assert key != answer_cache_key("What is CoT?", True, 4)
    assert key != answer_cache_key("What is CoT?", False, 3)
    assert normalize_question("Hello  World?!") == "hello world"


def test_new_index_version_misses_the_answer_cache():
    cache = AnswerCache(MemoryStore(), ttl_seconds=60)
    cache.put("q", True, 1, {"answer": "a"})
    assert cache.get("q", True, 1) == {"answer": "a"}
    assert cache.get("q", True, 2) is None


def test_memory_store_ttl_and_eviction():
    clock = {"now": 0.0}
    store = MemoryStore(max_items=2, clock=lambda: clock["now"])
    store.set("a", "1", ttl_seconds=10)
    store.set("b", "2")
    store.set("c", "3")
    assert store.get("a") is None
    assert store.get("b") == "2"
    store.set("d", "4", ttl_seconds=5)
    clock["now"] = 6.0
    assert store.get("d") is None


def test_rate_limit_minute_and_day_windows():
    clock = {"now": 1_000_000.0}
    limiter = RateLimiter(
        RateLimitConfig(o_enabled=True, per_minute=10, per_day=12), clock=lambda: clock["now"]
    )
    decision_list = [limiter.check("ip") for _ in range(11)]
    assert all(decision.o_allowed for decision in decision_list[:10])
    assert not decision_list[10].o_allowed and decision_list[10].window == "minute"
    assert limiter.check("other").o_allowed
    clock["now"] += 61
    assert limiter.check("ip").o_allowed
    assert not limiter.check("ip").o_allowed
    assert limiter.check("ip").window == "day"
    assert (
        RateLimiter(RateLimitConfig(o_enabled=False, per_minute=1, per_day=1)).check("x").o_allowed
    )


def test_sse_frame():
    assert sse_event("token", {"text": "hi"}) == 'event: token\ndata: {"text": "hi"}\n\n'


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Return (event, data) pairs of an SSE body."""
    event_list = []
    for frame in body.strip().split("\n\n"):
        line_list = frame.split("\n")
        event_list.append(
            (line_list[0][len("event: ") :], json.loads(line_list[1][len("data: ") :]))
        )
    return event_list


def _runtime(profile: str = "serving", collection_points: bool = True) -> Runtime:
    index_config = get_index_config()
    client = QdrantClient(":memory:")
    index_module.create_index_collection(client, "meridian_v3", 32, {})
    if collection_points:
        payload = {"chunk_id": "00000000-0000-0000-0000-000000000001", "arxiv_id": "p", "text": "t"}
        point_list = index_module.build_points(
            [payload],
            FakeDenseEncoder().encode_documents(["t"]),
            FakeSparseEncoder().encode_documents(["t"]),
        )
        index_module.upsert_points(client, "meridian_v3", point_list, 10)
    index_module.switch_alias(client, index_config.alias, "meridian_v3")
    retriever = FakeRetriever([CHUNKS])
    return Runtime(
        get_serving_config(profile),
        index_config,
        client,
        lambda: make_agent(Script(), retriever),
        retriever,
    )


def test_ask_streams_and_second_identical_question_hits_the_cache():
    runtime = _runtime()
    with TestClient(create_app(runtime)) as test_client:
        first = parse_sse(test_client.post("/v1/ask", json={"question": "What is CoT?"}).text)
        kind_list = [kind for kind, _ in first]
        assert kind_list[0] == "meta" and kind_list[-1] == "final" and "token" in kind_list
        final = first[-1][1]
        assert final["status"] == "answered"
        assert final["index_version"] == 3
        assert final["cached"] is False
        assert final["citations"][0]["url"].startswith("https://arxiv.org/abs/")
        second = parse_sse(test_client.post("/v1/ask", json={"question": "what is cot"}).text)
        assert second[-1][1]["cached"] is True
        follow = parse_sse(
            test_client.post(
                "/v1/ask", json={"question": "what is cot", "session_id": final["session_id"]}
            ).text
        )
        # a follow-up in a session with history is session-dependent and never served from cache
        assert follow[-1][1]["cached"] is False


def test_health_ready_index_metrics_and_search():
    runtime = _runtime(profile="demo")
    with TestClient(create_app(runtime)) as test_client:
        assert test_client.get("/healthz").json() == {"status": "ok"}
        assert test_client.get("/readyz").status_code == 200
        info = test_client.get("/v1/index").json()
        assert (info["index_version"], info["points"], info["papers"]) == (3, 1, 1)
        search = test_client.post("/v1/search", json={"query": "q", "k": 2}).json()
        assert len(search["chunks"]) == 2
        # the demo profile never returns full chunk text
        assert "text" not in search["chunks"][0]
        assert search["chunks"][0]["excerpt"]
        test_client.post("/v1/ask", json={"question": "What is CoT?"})
        metrics_text = test_client.get("/metrics").text
        assert "meridian_requests_total" in metrics_text
        assert "meridian_stage_latency_seconds" in metrics_text


def test_readyz_fails_on_an_empty_collection():
    with TestClient(create_app(_runtime(collection_points=False))) as test_client:
        assert test_client.get("/readyz").status_code == 503


def test_demo_rate_limit_returns_429_with_retry_after():
    runtime = _runtime(profile="demo")
    with TestClient(create_app(runtime)) as test_client:
        status_list = [
            test_client.post("/v1/search", json={"query": "q"}).status_code for _ in range(11)
        ]
        assert status_list[:10] == [200] * 10
        response = test_client.post("/v1/search", json={"query": "q"})
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) > 0
        too_long = test_client.post("/v1/ask", json={"question": "x" * 5000})
        assert too_long.status_code in {413, 429}
