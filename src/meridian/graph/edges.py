"""Conditional edge routing functions.

Each function reads the current state and returns the name of the next node (or
``END``). The three failure modes are routed independently: irrelevant
documents fall back to web search, an ungrounded answer retries generation, and
an off-target answer rewrites the query. All loops are bounded by
``max_iterations``.
"""

from langgraph.graph import END

from meridian.config import get_settings
from meridian.graph.state import GraphState


def decide_after_grading(state: GraphState) -> str:
    """Route to generation if any document passed grading, else to web search."""
    if state.get("graded_docs"):
        return "generate"
    return "web_search"


def decide_after_hallucination(state: GraphState) -> str:
    """Route the hallucination outcome.

    Grounded answers proceed to answer grading. Ungrounded answers retry
    generation until the iteration cap is reached, after which the graph ends
    with the best available answer.
    """
    settings = get_settings()
    if state.get("hallucination_score") == "no":
        return "check_answer"
    # The generation is ungrounded; retry unless the cap has been reached.
    if state.get("iteration_count", 0) >= settings.max_iterations:
        return END
    return "generate"


def decide_after_answer(state: GraphState) -> str:
    """Route the answer-relevance outcome.

    A relevant answer ends the graph. An off-target answer rewrites the query
    and re-retrieves until the iteration cap is reached.
    """
    settings = get_settings()
    if state.get("answer_score") == "yes":
        return END
    if state.get("iteration_count", 0) >= settings.max_iterations:
        return END
    return "rewrite_query"
