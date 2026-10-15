"""`rewrite` builds a standalone query, `retrieve` runs hybrid search plus rerank, `grade` keeps
chunks by reranker score bands and sends only the uncertain band to the LLM grader, and
`decide` chooses between generating, retrying, web fallback, and declining."""

import logging
import time

from meridian.agent.deps import AgentDeps, timed
from meridian.agent.prompts import load_prompt
from meridian.agent.state import (
    DECISION_GENERATE,
    DECISION_INSUFFICIENT,
    DECISION_RETRY,
    DECISION_WEB,
    AgentState,
    format_history,
    recent_history,
)

logger = logging.getLogger(__name__)

PAYLOAD_KEY_LIST = (
    "arxiv_id",
    "version",
    "title",
    "section_title",
    "section_index",
    "page_start",
    "page_end",
    "text",
    "char_start",
    "char_end",
    "index_version",
)


def make_rewrite(deps: AgentDeps):
    """Return the query rewriter node."""

    def rewrite(state: AgentState) -> dict:
        started = time.perf_counter()
        tried_list = list(state.get("tried_queries") or [])
        retry_instruction = ""
        if tried_list:
            retry_instruction = (
                "These earlier queries found too little support; write a different query that "
                "uses other terms for the same need: " + "; ".join(tried_list)
            )
        system = load_prompt("rewrite").format(
            topic_name=deps.topic_name, retry_instruction=retry_instruction
        )
        history = format_history(
            recent_history(state, deps.agent_config.conversation.history_messages)
        )
        user = f"Conversation:\n{history or '(none)'}\n\nLatest message: {state['question']}"
        try:
            reply = deps.llm.chat_json(
                "rewriter",
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            query = str(reply.get("query", "")).strip()
        except ValueError as exc:
            logger.warning("rewriter reply unusable", extra={"error": str(exc)})
            query = ""
        if not query:
            query = state["question"]
        return {
            "standalone_query": query,
            "tried_queries": tried_list + [query],
            "timings": timed(state, "rewrite", started),
        }

    return rewrite


def make_retrieve(deps: AgentDeps):
    """Return the retrieval node."""

    def retrieve(state: AgentState) -> dict:
        started = time.perf_counter()
        result = deps.retriever.search(
            state["standalone_query"],
            mode=deps.agent_config.retrieval_mode,
            k=deps.final_k,
            paper_filter=state.get("pinned_papers") or None,
        )
        retrieved_list = [
            {
                "chunk_id": chunk.chunk_id,
                "score": chunk.rerank_score if chunk.rerank_score is not None else chunk.score,
                "payload": {key: chunk.payload.get(key) for key in PAYLOAD_KEY_LIST},
            }
            for chunk in result.chunk_list
        ]
        timing_dict = timed(state, "retrieve", started)
        for stage, seconds in result.timing_dict.items():
            timing_dict[f"retrieval_{stage}"] = timing_dict.get(f"retrieval_{stage}", 0.0) + seconds
        return {"retrieved": retrieved_list, "timings": timing_dict}

    return retrieve


def llm_grade(deps: AgentDeps, question: str, text: str) -> bool:
    """Ask the grader whether a passage helps answer the question."""
    try:
        reply = deps.llm.chat_json(
            "grader",
            [
                {"role": "system", "content": load_prompt("grade")},
                {"role": "user", "content": f"Question: {question}\n\nPassage:\n{text}"},
            ],
        )
    except ValueError as exc:
        logger.warning("grader reply unusable, dropping chunk", extra={"error": str(exc)})
        return False
    return reply.get("relevant") is True


def make_grade(deps: AgentDeps):
    """Return the score-band grading node."""

    def grade(state: AgentState) -> dict:
        started = time.perf_counter()
        config = deps.agent_config.grading
        stats = {"accepted_by_score": 0, "rejected_by_score": 0, "llm_graded": 0, "llm_kept": 0}
        graded_list: list[dict] = []
        for chunk in state.get("retrieved", []):
            score = float(chunk["score"])
            if config.o_score_band and score >= config.accept_score:
                stats["accepted_by_score"] += 1
                graded_list.append(chunk)
                continue
            if config.o_score_band and score < config.reject_score:
                stats["rejected_by_score"] += 1
                continue
            stats["llm_graded"] += 1
            if llm_grade(deps, state["standalone_query"], chunk["payload"]["text"]):
                stats["llm_kept"] += 1
                graded_list.append(chunk)
        previous = state.get("grade_stats") or {}
        merged = {key: previous.get(key, 0) + value for key, value in stats.items()}
        return {
            "graded": graded_list,
            "grade_stats": merged,
            "timings": timed(state, "grade", started),
        }

    return grade


def make_decide(deps: AgentDeps):
    """Return the node that decides how to proceed after grading."""

    def decide(state: AgentState) -> dict:
        config = deps.agent_config.decide
        supporting = len(state.get("graded", []))
        if supporting >= config.min_supporting_chunks:
            return {"decision": DECISION_GENERATE}
        if state.get("retries", 0) < config.max_retries:
            return {"decision": DECISION_RETRY, "retries": state.get("retries", 0) + 1}
        if config.o_web_fallback and deps.web is not None:
            return {"decision": DECISION_WEB, "fallback": "web"}
        return {"decision": DECISION_INSUFFICIENT, "fallback": "insufficient"}

    return decide


def route_after_decide(state: AgentState) -> str:
    """Conditional edge after `decide`."""
    return {
        DECISION_GENERATE: "generate",
        DECISION_RETRY: "rewrite",
        DECISION_WEB: "web_search",
        DECISION_INSUFFICIENT: "insufficient",
    }[state["decision"]]


def make_web_search(deps: AgentDeps):
    """Return the web fallback node."""

    def web_search(state: AgentState) -> dict:
        started = time.perf_counter()
        assert deps.web is not None
        result_list = deps.web.search(state["standalone_query"], deps.agent_config.web.max_results)
        return {"web_results": result_list, "timings": timed(state, "web_search", started)}

    return web_search


def route_after_web(state: AgentState) -> str:
    """Generate when the web returned anything or some corpus chunk survived, else decline."""
    if state.get("web_results") or state.get("graded"):
        return "generate"
    return "insufficient"
