"""GraphState schema for the agentic RAG state machine.

The state accumulates as the graph executes. ``iteration_count`` uses an
additive reducer so that any node returning ``{"iteration_count": 1}``
increments the running total rather than overwriting it; this bounds the two
recovery loops (generation retry and query rewrite) against ``max_iterations``.
"""

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    """Mutable state threaded through every node of the graph."""

    query: str  # original user query
    session_id: str  # cross-session memory key, distinct from the checkpoint thread id
    conversation_history: str  # prior turns for this session, recalled at graph start
    rewritten_query: str  # query after the rewrite node
    query_type: str  # "factual" | "comparative" | "methodological"
    retrieved_docs: list[dict]  # post-RRF, pre-cross-encoder candidates
    graded_docs: list[dict]  # cross-encoder filtered documents (score >= 0.5)
    generation: str  # generator output
    hallucination_score: str  # "yes" | "no"
    answer_score: str  # "yes" | "no"
    web_search_result: str  # Tavily result if CRAG fallback triggered
    source: str  # "corpus" | "web"
    iteration_count: Annotated[int, operator.add]  # accumulates via add reducer
