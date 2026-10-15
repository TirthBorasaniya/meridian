"""`start_turn` resets per-turn state; `route` classifies the request; `respond_without_search`
handles clarification and out-of-scope refusals."""

import logging
import time

from meridian.agent.deps import AgentDeps, timed
from meridian.agent.prompts import load_prompt
from meridian.agent.state import (
    ROUTE_ANSWER,
    ROUTE_CLARIFY,
    ROUTE_OUT_OF_SCOPE,
    STATUS_CLARIFY,
    STATUS_OUT_OF_SCOPE,
    AgentState,
    format_history,
    recent_history,
)

logger = logging.getLogger(__name__)

ROUTE_SET = {ROUTE_ANSWER, ROUTE_CLARIFY, ROUTE_OUT_OF_SCOPE}


def make_route(deps: AgentDeps):
    """Return the router node."""

    def route(state: AgentState) -> dict:
        started = time.perf_counter()
        system = load_prompt("route").format(
            topic_name=deps.topic_name, topic_description=deps.topic_description
        )
        history = format_history(
            recent_history(state, deps.agent_config.conversation.history_messages)
        )
        user = f"Conversation:\n{history or '(none)'}\n\nLatest message: {state['question']}"
        try:
            reply = deps.llm.chat_json(
                "router", [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
            route_name = str(reply.get("route", ROUTE_ANSWER))
            clarifying = str(reply.get("clarifying_question", "")).strip()
        except ValueError as exc:
            logger.warning("router reply unusable, answering", extra={"error": str(exc)})
            route_name, clarifying = ROUTE_ANSWER, ""
        if route_name not in ROUTE_SET:
            route_name = ROUTE_ANSWER
        if route_name == ROUTE_CLARIFY and not clarifying:
            clarifying = "Could you say more precisely what you would like to know?"
        return {
            "route": route_name,
            "clarifying_question": clarifying,
            "timings": timed(state, "route", started),
        }

    return route


def make_respond_without_search(deps: AgentDeps):
    """Return the node that answers clarify and out-of-scope routes without retrieval."""

    def respond(state: AgentState) -> dict:
        if state.get("route") == ROUTE_CLARIFY:
            return {"answer": state["clarifying_question"], "status": STATUS_CLARIFY}
        refusal = (
            f"This assistant answers questions about {deps.topic_name} from a corpus of arXiv "
            "papers, so it cannot help with this request."
        )
        return {"answer": refusal, "status": STATUS_OUT_OF_SCOPE}

    return respond


def route_after_router(state: AgentState) -> str:
    """Conditional edge after `route`."""
    return "rewrite" if state.get("route") == ROUTE_ANSWER else "respond_without_search"
