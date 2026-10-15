"""Unit tests for the agent graph: routing, score-band grading, retries and fallback, citation
verification, rendering, and the per-turn state reset that fixes the sticky web-fallback bug."""

import json

import pytest

from meridian.agent.state import INSUFFICIENT_TEXT, PER_TURN_DEFAULTS, per_turn_reset
from tests.agent_fakes import FakeRetriever, FakeWeb, Script, make_agent, make_chunk

pytestmark = pytest.mark.unit

GOOD_CHUNKS = [make_chunk(1, 0.95), make_chunk(2, 0.9), make_chunk(3, 0.85)]
WEAK_CHUNKS = [make_chunk(4, 0.4), make_chunk(5, 0.1)]


def test_answer_with_resolved_citations_and_rendering():
    agent = make_agent(Script(), FakeRetriever([GOOD_CHUNKS]))
    result = agent.ask("What is chain-of-thought prompting?", "s1")
    assert result.status == "answered"
    first_id = GOOD_CHUNKS[0].chunk_id
    assert f"[c:{first_id}]" in result.answer
    citation = result.citations[0]
    assert citation["url"] == "https://arxiv.org/abs/2201.11903v6"
    assert citation["section_title"] == "1 Section"
    assert citation["excerpt"] == "Fact number 1 is stated here. A second sentence."
    assert citation["external"] is False
    assert result.verification["n_judged"] == 2
    assert result.contexts and result.fallback == ""


def test_score_band_sends_only_the_middle_to_the_llm():
    script = Script()
    chunk_list = [make_chunk(1, 0.95), make_chunk(2, 0.5), make_chunk(3, 0.1)]
    agent = make_agent(script, FakeRetriever([chunk_list]))
    result = agent.ask("q", "s1")
    assert result.grade_stats == {
        "accepted_by_score": 1,
        "rejected_by_score": 1,
        "llm_graded": 1,
        "llm_kept": 1,
    }
    assert script.count_dict["grade"] == 1


def test_without_score_band_every_chunk_is_llm_graded():
    script = Script()
    chunk_list = [make_chunk(1, 0.95), make_chunk(2, 0.5), make_chunk(3, 0.1)]
    agent = make_agent(script, FakeRetriever([chunk_list]), grading={"o_score_band": False})
    result = agent.ask("q", "s1")
    assert result.grade_stats["llm_graded"] == 3
    assert script.count_dict["grade"] == 3


def test_retries_rewrite_then_decline_without_web():
    script = Script(grade=json.dumps({"relevant": False}))
    retriever = FakeRetriever([WEAK_CHUNKS])
    agent = make_agent(script, retriever, decide={"o_web_fallback": False})
    result = agent.ask("q", "s1")
    assert result.status == "insufficient"
    assert result.answer == INSUFFICIENT_TEXT
    assert result.retries == 2
    assert len(retriever.call_list) == 3
    assert script.count_dict["rewrite"] == 3
    assert "generate" not in script.count_dict


def test_sticky_web_fallback_does_not_leak_into_the_next_turn():
    script = Script(
        grade=json.dumps({"relevant": False}),
        generate=lambda user: (
            "Web says so [w:1]." if "[w:1]" in user else "Fact one [c:1]. Fact two [c:2]."
        ),
    )
    web = FakeWeb()
    retriever = FakeRetriever([WEAK_CHUNKS, WEAK_CHUNKS, WEAK_CHUNKS, GOOD_CHUNKS])
    agent = make_agent(script, retriever, web=web)
    first = agent.ask("first question", "shared")
    assert first.fallback == "web"
    assert first.citations[0]["external"] is True
    second = agent.ask("second question", "shared")
    assert second.fallback == ""
    assert second.status == "answered"
    assert all(citation["kind"] == "corpus" for citation in second.citations)
    assert web.call_count == 1
    state = agent.graph.get_state(agent._config("shared")).values
    assert state["web_results"] == []
    assert len(state["messages"]) == 4


def test_out_of_scope_is_refused_without_retrieval():
    script = Script(route=json.dumps({"route": "out_of_scope", "clarifying_question": ""}))
    retriever = FakeRetriever([GOOD_CHUNKS])
    result = make_agent(script, retriever).ask("Best sourdough recipe?", "s1")
    assert result.status == "out_of_scope"
    assert "LLM reasoning and evaluation" in result.answer
    assert retriever.call_list == []


def test_clarify_route_asks_back():
    script = Script(route=json.dumps({"route": "clarify", "clarifying_question": "Which method?"}))
    result = make_agent(script, FakeRetriever([GOOD_CHUNKS])).ask("it?", "s1")
    assert (result.status, result.answer) == ("clarify", "Which method?")


def test_uncited_sentence_is_removed_and_invalid_alias_is_recorded():
    script = Script(
        generate="Fact one holds [c:1]. Fact two holds [c:2]. A filler remark. Fact three [c:9]."
    )
    result = make_agent(script, FakeRetriever([GOOD_CHUNKS])).ask("q", "s1")
    assert result.status == "answered"
    assert "filler" not in result.answer
    assert "Fact three" not in result.answer
    assert result.verification["invalid_citations"] == ["[c:9]"]
    assert result.verification["removed"] == 2


def test_mostly_unsupported_answer_is_regenerated_once_then_declined():
    script = Script(verify=json.dumps({"supported": False}))
    result = make_agent(script, FakeRetriever([GOOD_CHUNKS])).ask("q", "s1")
    assert script.count_dict["generate"] == 2
    assert result.regenerations == 1
    assert result.status == "insufficient"


def test_sampled_verification_judges_a_subset():
    script = Script()
    result = make_agent(script, FakeRetriever([GOOD_CHUNKS]), verify={"sample_rate": 0.0}).ask(
        "q", "s1"
    )
    assert result.verification["n_judged"] == 0
    assert "verify" not in script.count_dict


def test_pinned_papers_filter_and_persist():
    retriever = FakeRetriever([GOOD_CHUNKS])
    agent = make_agent(Script(), retriever)
    agent.ask("q1", "s1", pinned_papers=["2201.11903"])
    agent.ask("q2", "s1")
    assert [call["paper_filter"] for call in retriever.call_list] == [
        ["2201.11903"],
        ["2201.11903"],
    ]


def test_per_turn_reset_clears_every_per_turn_field():
    update = per_turn_reset("new")
    assert set(PER_TURN_DEFAULTS) <= set(update)
    assert update["fallback"] == "" and update["web_results"] == [] and update["graded"] == []
    assert "pinned_papers" not in update
