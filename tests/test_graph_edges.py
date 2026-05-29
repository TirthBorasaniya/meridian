"""Unit tests for the conditional edge routing logic."""

from langgraph.graph import END

from meridian.graph.edges import (
    decide_after_answer,
    decide_after_grading,
    decide_after_hallucination,
)


def test_grading_routes_to_generate_when_docs_present():
    assert decide_after_grading({"graded_docs": [{"chunk_id": "a"}]}) == "generate"


def test_grading_routes_to_web_when_no_docs():
    assert decide_after_grading({"graded_docs": []}) == "web_search"
    assert decide_after_grading({}) == "web_search"


def test_hallucination_grounded_goes_to_check_answer():
    state = {"hallucination_score": "no", "iteration_count": 0}
    assert decide_after_hallucination(state) == "check_answer"


def test_hallucination_grounded_checks_answer_even_at_cap():
    # A grounded answer proceeds to answer grading regardless of iteration count.
    state = {"hallucination_score": "no", "iteration_count": 3}
    assert decide_after_hallucination(state) == "check_answer"


def test_hallucination_ungrounded_retries_generate_under_cap():
    state = {"hallucination_score": "yes", "iteration_count": 1}
    assert decide_after_hallucination(state) == "generate"


def test_hallucination_ungrounded_ends_at_cap():
    state = {"hallucination_score": "yes", "iteration_count": 3}
    assert decide_after_hallucination(state) == END


def test_answer_relevant_ends():
    assert decide_after_answer({"answer_score": "yes", "iteration_count": 0}) == END


def test_answer_offtarget_rewrites_under_cap():
    state = {"answer_score": "no", "iteration_count": 1}
    assert decide_after_answer(state) == "rewrite_query"


def test_answer_offtarget_ends_at_cap():
    state = {"answer_score": "no", "iteration_count": 3}
    assert decide_after_answer(state) == END
