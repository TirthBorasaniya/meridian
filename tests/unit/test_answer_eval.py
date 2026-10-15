"""Unit tests for the E2 helpers: citation precision, unsupported rate, samples, and the bridge."""

import json
from types import SimpleNamespace

import pytest

from meridian.common.config import get_models_config
from meridian.common.llm_client import LLMClient, MemoryResponseCache
from meridian.eval.answer_eval import (
    citation_pairs,
    citation_precision,
    ragas_samples,
    require_references,
    unsupported_sentence_rate,
)
from meridian.eval.datasets import load_answer_set, load_out_of_scope_set
from meridian.eval.ragas_bridge import CachedJudgeChatModel
from tests.fakes import ScriptedTransport

pytestmark = pytest.mark.unit

GRADED = [
    {"chunk_id": "a", "payload": {"text": "Self-consistency votes over sampled paths."}},
    {"chunk_id": "b", "payload": {"text": "Calibration compares confidence and accuracy."}},
]


def _result(answer: str, status: str = "answered", verification=None):
    return SimpleNamespace(
        answer=answer,
        status=status,
        graded=GRADED,
        contexts=["ctx"],
        verification=verification or {},
    )


def test_citation_pairs_and_precision():
    answer = "Votes are taken [c:a]. Confidence is compared [c:a][c:b]."
    assert len(citation_pairs(answer, GRADED)) == 3

    def responder(model, message_list, params):
        user = message_list[-1]["content"]
        supported = ("Votes" in user and "Self-consistency" in user) or (
            "Confidence" in user and "Calibration" in user
        )
        return json.dumps({"supported": supported})

    llm = LLMClient(get_models_config(), ScriptedTransport(responder), MemoryResponseCache())
    precision = citation_precision(llm, [_result(answer), _result("x [c:a].", "insufficient")])
    assert precision == {"citation_precision": 2 / 3, "pairs_supported": 2, "pairs_total": 3}


def test_unsupported_rate_and_samples():
    result_list = [
        _result("a", verification={"n_sentences": 4, "unsupported": [{}]}),
        _result("b", verification={"n_sentences": 6, "unsupported": [{}, {}]}),
    ]
    assert unsupported_sentence_rate(result_list)["unsupported_sentence_rate"] == pytest.approx(0.3)
    sample = ragas_samples([{"question": "q", "reference": "r"}], [_result("Fact [c:a].")])[0]
    assert sample == {
        "user_input": "q",
        "response": "Fact.",
        "retrieved_contexts": ["ctx"],
        "reference": "r",
    }


def test_committed_sets_have_the_spec_sizes():
    assert len(load_answer_set()) == 75
    assert len(load_answer_set(split="fast")) == 20
    assert len(load_out_of_scope_set()) == 25
    with pytest.raises(RuntimeError, match="lack references"):
        require_references(load_answer_set())


def test_judge_bridge_routes_through_the_cached_client():
    transport = ScriptedTransport(lambda *_: "verdict")
    llm = LLMClient(get_models_config(), transport, MemoryResponseCache())
    model = CachedJudgeChatModel(llm=llm)
    assert model.invoke("Is it supported?").content == "verdict"
    assert model.invoke("Is it supported?").content == "verdict"
    assert len(transport.request_list) == 1
    assert transport.request_list[0][0] == get_models_config().role("judge").model
