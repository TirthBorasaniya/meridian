"""Unit tests for the E3 measurement helpers with a scripted runtime and a fake semantic cache."""

import numpy as np
import pytest

from meridian.eval import serving_eval
from meridian.eval.datasets import read_jsonl
from tests.unit.test_serving import _runtime

pytestmark = pytest.mark.unit


def test_answer_run_and_cache_hits(monkeypatch):
    from meridian.common.config import PriceConfig, get_models_config

    # the judge is unpriced in configs; a test-only price lets the cost arithmetic run
    judge_model = get_models_config().role("judge").model
    monkeypatch.setitem(
        get_models_config().prices_usd_per_million_tokens,
        judge_model,
        PriceConfig(input=1.0, output=1.0),
    )
    runtime = _runtime()
    question_list = ["What is CoT?", "How is calibration error computed?"]
    summary = serving_eval.answer_run(runtime, question_list)
    assert summary["n"] == 2
    assert summary["llm_calls_per_answer"] > 0
    assert summary["cost_usd_per_answer"] > 0
    assert summary["latency"]["count"] == 2
    cache_summary = serving_eval.answer_cache_run(runtime, question_list)
    assert cache_summary["lookups"] == 4
    assert cache_summary["hit_rate"] == 1.0


class FakeSemanticCache:
    """Cosine similarity over the fake encoder, with the same interface as SemanticAnswerCache."""

    def __init__(self, redis_url, dense_encoder, distance_threshold, ttl_seconds, name):
        self.encoder = dense_encoder
        self.threshold = distance_threshold
        self.entry_list: list[tuple[np.ndarray, int, dict]] = []

    def clear(self):
        self.entry_list = []

    def store(self, question, index_version, payload):
        self.entry_list.append(
            (np.array(self.encoder.encode_query(question)), index_version, payload)
        )

    def lookup(self, question, index_version):
        vector = np.array(self.encoder.encode_query(question))
        for stored, version, payload in self.entry_list:
            distance = 1.0 - float(stored @ vector)
            if version == index_version and distance <= self.threshold:
                return {**payload, "semantic_match": {"distance": distance}}
        return None


def test_semantic_run_counts_wrong_hits(monkeypatch):
    from meridian.serving import cache as cache_module
    from tests.fakes import FakeDenseEncoder

    monkeypatch.setattr(cache_module, "SemanticAnswerCache", FakeSemanticCache)
    runtime = _runtime()
    runtime.retriever.dense_encoder = FakeDenseEncoder()
    probe_list = [
        {"pid": "p1", "cached_question": "alpha beta", "probe": "alpha beta", "same_intent": True},
        {
            "pid": "p2",
            "cached_question": "alpha beta",
            "probe": "gamma delta",
            "same_intent": False,
        },
        {"pid": "p3", "cached_question": "alpha beta", "probe": "beta alpha", "same_intent": False},
    ]
    summary = serving_eval.semantic_cache_run(runtime, probe_list, 0.1, "redis://unused")
    assert summary["same_intent_hit_rate"] == 1.0
    assert summary["wrong_hit_rate"] == 0.5


def test_committed_probe_set_is_balanced():
    probe_list = read_jsonl("eval/cache_probes.jsonl")
    assert len(probe_list) == 30
    assert sum(probe["same_intent"] for probe in probe_list) == 15
