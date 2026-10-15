"""`generate` writes a cited answer from the graded chunks and web results, streaming tokens;
`verify` drops invalid citations, judges cited sentences on a sampled basis, and either removes
unsupported sentences or asks for one regeneration; `insufficient` and `finalize` render the
turn's answer and citations."""

import hashlib
import logging
import time

from meridian.agent.citations import (
    cited_chunk_ids,
    cited_web_indices,
    excerpt,
    extract_citations,
    resolve_aliases,
    split_sentences,
    strip_citations,
)
from meridian.agent.deps import AgentDeps, stream_writer, timed
from meridian.agent.prompts import load_prompt
from meridian.agent.state import (
    INSUFFICIENT_TEXT,
    STATUS_ANSWERED,
    STATUS_INSUFFICIENT,
    VERIFY_ACCEPT,
    VERIFY_REGENERATE,
    AgentState,
)

logger = logging.getLogger(__name__)


# ============= Context =============


def build_sources(state: AgentState) -> tuple[str, dict[str, str], list[str]]:
    """
    Render the numbered sources for the generator.

    Returns
    -------
    sources, alias_dict, context_list : tuple
        Source block, alias to chunk id map, and the raw texts used as answer contexts.
    """
    block_list: list[str] = []
    alias_dict: dict[str, str] = {}
    context_list: list[str] = []
    for index, chunk in enumerate(state.get("graded", []), start=1):
        payload = chunk["payload"]
        alias_dict[str(index)] = chunk["chunk_id"]
        context_list.append(payload["text"])
        block_list.append(
            f"[c:{index}] {payload.get('title', '')} | section: {payload.get('section_title', '')}"
            f"\n{payload['text']}"
        )
    for index, result in enumerate(state.get("web_results", []), start=1):
        context_list.append(result["content"])
        block_list.append(
            f"[w:{index}] external web source: {result['title']} ({result['url']})\n"
            f"{result['content']}"
        )
    return "\n\n".join(block_list), alias_dict, context_list


# ============= Generate =============


def make_generate(deps: AgentDeps):
    """Return the generation node."""

    def generate(state: AgentState) -> dict:
        started = time.perf_counter()
        sources, alias_dict, context_list = build_sources(state)
        system = load_prompt("generate").format(topic_name=deps.topic_name)
        user = f"Question: {state['question']}\n\nSources:\n{sources}"
        feedback = state.get("verify_feedback") or ""
        if feedback:
            user += (
                "\n\nA previous draft contained sentences the sources do not support:\n"
                f"{feedback}\nWrite the answer again, keeping only statements the cited sources "
                "support."
            )
        writer = stream_writer()
        if feedback:
            writer({"type": "reset"})
        response = deps.llm.chat(
            "generator",
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            on_delta=lambda delta: writer({"type": "token", "text": delta}),
        )
        draft, invalid_list = resolve_aliases(
            response.text, alias_dict, len(state.get("web_results", []))
        )
        return {
            "draft": draft.strip(),
            "alias_dict": alias_dict,
            "invalid_citations": list(state.get("invalid_citations") or []) + invalid_list,
            "contexts": context_list,
            "regenerations": state.get("regenerations", 0) + (1 if feedback else 0),
            "timings": timed(state, "generate", started),
        }

    return generate


# ============= Verify =============


def sampled(seed: int, question: str, sentence: str, rate: float) -> bool:
    """Deterministic Bernoulli draw per sentence, so repeated runs judge the same sentences."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(f"{seed}|{question}|{sentence}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def judge_support(deps: AgentDeps, sentence: str, passage_list: list[str]) -> bool:
    """Ask the judge whether the cited passages support a sentence."""
    passages = "\n\n".join(
        f"Passage {index}:\n{text}" for index, text in enumerate(passage_list, 1)
    )
    try:
        reply = deps.llm.chat_json(
            "judge",
            [
                {"role": "system", "content": load_prompt("verify")},
                {"role": "user", "content": f"Sentence: {sentence}\n\n{passages}"},
            ],
        )
    except ValueError as exc:
        logger.warning("judge reply unusable, treating as unsupported", extra={"error": str(exc)})
        return False
    return reply.get("supported") is True


def make_verify(deps: AgentDeps):
    """Return the citation verification node."""

    def verify(state: AgentState) -> dict:
        started = time.perf_counter()
        config = deps.agent_config.verify
        draft = state.get("draft", "")
        chunk_text_dict = {
            chunk["chunk_id"]: chunk["payload"]["text"] for chunk in state.get("graded", [])
        }
        web_text_dict = {
            str(index): result["content"]
            for index, result in enumerate(state.get("web_results", []), start=1)
        }
        sentence_list = split_sentences(draft)
        if not sentence_list or strip_citations(draft).strip() == INSUFFICIENT_TEXT:
            return {
                "verify_decision": VERIFY_ACCEPT,
                "answer": INSUFFICIENT_TEXT,
                "status": STATUS_INSUFFICIENT,
                "verification": {"n_sentences": 0, "unsupported": []},
                "timings": timed(state, "verify", started),
            }
        unsupported_list: list[dict] = []
        judged = 0
        for sentence in sentence_list:
            passage_list = [
                chunk_text_dict[key] if kind == "c" else web_text_dict[key]
                for kind, key in extract_citations(sentence)
                if (kind == "c" and key in chunk_text_dict)
                or (kind == "w" and key in web_text_dict)
            ]
            if not passage_list:
                unsupported_list.append({"sentence": sentence, "reason": "uncited"})
                continue
            if not sampled(config.sample_seed, state["question"], sentence, config.sample_rate):
                continue
            judged += 1
            if not judge_support(deps, strip_citations(sentence), passage_list):
                unsupported_list.append({"sentence": sentence, "reason": "judge"})
        unsupported_fraction = len(unsupported_list) / len(sentence_list)
        verification = {
            "n_sentences": len(sentence_list),
            "n_judged": judged,
            "unsupported": unsupported_list,
            "unsupported_fraction": unsupported_fraction,
            "invalid_citations": list(state.get("invalid_citations") or []),
            "sample_rate": config.sample_rate,
        }
        if (
            unsupported_fraction > config.regenerate_above_unsupported_fraction
            and state.get("regenerations", 0) < config.max_regenerations
        ):
            return {
                "verify_decision": VERIFY_REGENERATE,
                "verify_feedback": "\n".join(
                    f"- {strip_citations(item['sentence'])}" for item in unsupported_list
                ),
                "verification": verification,
                "timings": timed(state, "verify", started),
            }
        unsupported_set = {item["sentence"] for item in unsupported_list}
        kept_list = [sentence for sentence in sentence_list if sentence not in unsupported_set]
        verification["removed"] = len(sentence_list) - len(kept_list)
        if not kept_list:
            return {
                "verify_decision": VERIFY_ACCEPT,
                "answer": INSUFFICIENT_TEXT,
                "status": STATUS_INSUFFICIENT,
                "verification": verification,
                "timings": timed(state, "verify", started),
            }
        return {
            "verify_decision": VERIFY_ACCEPT,
            "answer": " ".join(kept_list),
            "status": STATUS_ANSWERED,
            "verification": verification,
            "timings": timed(state, "verify", started),
        }

    return verify


def route_after_verify(state: AgentState) -> str:
    """Conditional edge after `verify`."""
    return "generate" if state.get("verify_decision") == VERIFY_REGENERATE else "finalize"


# ============= Finalize =============


def insufficient(state: AgentState) -> dict:
    """Decline when the corpus does not support an answer."""
    return {"answer": INSUFFICIENT_TEXT, "status": STATUS_INSUFFICIENT}


def make_finalize(deps: AgentDeps):
    """Return the node that renders citations and records the assistant message."""

    def finalize(state: AgentState) -> dict:
        answer = state.get("answer") or INSUFFICIENT_TEXT
        status = state.get("status") or STATUS_INSUFFICIENT
        chunk_dict = {chunk["chunk_id"]: chunk for chunk in state.get("graded", [])}
        citation_list: list[dict] = []
        for chunk_id in cited_chunk_ids(answer):
            chunk = chunk_dict.get(chunk_id)
            if chunk is None:
                continue
            payload = chunk["payload"]
            versioned_id = f"{payload['arxiv_id']}v{payload['version']}"
            citation_list.append(
                {
                    "kind": "corpus",
                    "marker": f"c:{chunk_id}",
                    "chunk_id": chunk_id,
                    "arxiv_id": payload["arxiv_id"],
                    "version": payload["version"],
                    "title": payload.get("title", ""),
                    "section_title": payload.get("section_title", ""),
                    "url": f"https://arxiv.org/abs/{versioned_id}",
                    "excerpt": excerpt(payload["text"], deps.agent_config.excerpt_sentences),
                    "external": False,
                }
            )
        web_list = state.get("web_results", [])
        for index in cited_web_indices(answer):
            if 1 <= index <= len(web_list):
                result = web_list[index - 1]
                citation_list.append(
                    {
                        "kind": "web",
                        "marker": f"w:{index}",
                        "title": result["title"],
                        "url": result["url"],
                        "excerpt": excerpt(result["content"], deps.agent_config.excerpt_sentences),
                        "external": True,
                    }
                )
        return {
            "answer": answer,
            "status": status,
            "citations": citation_list,
            "messages": [{"role": "assistant", "content": strip_citations(answer)}],
        }

    return finalize
