"""Agent state. Only the conversation messages and the user-pinned paper filter persist across
turns; every other field is per-turn and is reset by the first node of each turn, which is
what prevents a previous turn's web fallback or grades from leaking into the next one.
"""

import operator
from copy import deepcopy
from typing import Annotated, TypedDict

# ============= Constants =============

ROUTE_ANSWER = "answer"
ROUTE_CLARIFY = "clarify"
ROUTE_OUT_OF_SCOPE = "out_of_scope"
STATUS_ANSWERED = "answered"
STATUS_INSUFFICIENT = "insufficient"
STATUS_CLARIFY = "clarify"
STATUS_OUT_OF_SCOPE = "out_of_scope"
DECISION_GENERATE = "generate"
DECISION_RETRY = "retry"
DECISION_WEB = "web"
DECISION_INSUFFICIENT = "insufficient"
VERIFY_ACCEPT = "accept"
VERIFY_REGENERATE = "regenerate"
INSUFFICIENT_TEXT = "The corpus does not contain enough support to answer this question."


class AgentState(TypedDict, total=False):
    """State of one conversation thread."""

    # persistent across turns
    messages: Annotated[list[dict], operator.add]
    pinned_papers: list[str]
    # per turn
    question: str
    route: str
    clarifying_question: str
    standalone_query: str
    tried_queries: list[str]
    retries: int
    retrieved: list[dict]
    graded: list[dict]
    grade_stats: dict
    decision: str
    fallback: str
    web_results: list[dict]
    alias_dict: dict[str, str]
    draft: str
    invalid_citations: list[str]
    verification: dict
    verify_decision: str
    regenerations: int
    verify_feedback: str
    answer: str
    status: str
    citations: list[dict]
    contexts: list[str]
    timings: dict[str, float]


PER_TURN_DEFAULTS: dict = {
    "question": "",
    "route": "",
    "clarifying_question": "",
    "standalone_query": "",
    "tried_queries": [],
    "retries": 0,
    "retrieved": [],
    "graded": [],
    "grade_stats": {},
    "decision": "",
    "fallback": "",
    "web_results": [],
    "alias_dict": {},
    "draft": "",
    "invalid_citations": [],
    "verification": {},
    "verify_decision": "",
    "regenerations": 0,
    "verify_feedback": "",
    "answer": "",
    "status": "",
    "citations": [],
    "contexts": [],
    "timings": {},
}


def per_turn_reset(question: str) -> dict:
    """Return the update that clears every per-turn field and records the new question."""
    update = deepcopy(PER_TURN_DEFAULTS)
    update["question"] = question
    update["messages"] = [{"role": "user", "content": question}]
    return update


def recent_history(state: AgentState, max_messages: int) -> list[dict]:
    """Return earlier messages of the thread, excluding the current question."""
    message_list = list(state.get("messages", []))
    if message_list and message_list[-1].get("role") == "user":
        message_list = message_list[:-1]
    return message_list[-max_messages:] if max_messages else []


def format_history(message_list: list[dict]) -> str:
    """Render messages as `role: content` lines."""
    return "\n".join(f"{message['role']}: {message['content']}" for message in message_list)
