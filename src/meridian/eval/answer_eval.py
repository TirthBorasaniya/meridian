"""E2: answer quality on the answer set A and refusals on the out-of-scope set O.

Metrics: RAGAS `Faithfulness`, `ResponseRelevancy`, `LLMContextPrecisionWithReference`, and
`LLMContextRecall` over the contexts the agent actually used; citation precision (the share of
sentence-citation pairs whose cited chunk supports the sentence, judged pair by pair); the
unsupported-sentence rate found by the verifier on drafts; and refusal accuracy on O. The judge
role is never the generator. Writes `results/e2_answers.json`.
"""

import logging
import math
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from meridian.agent.citations import extract_citations, split_sentences, strip_citations
from meridian.common.config import (
    IndexConfig,
    VersionedConfig,
    get_index_config,
    get_models_config,
    load_config,
)
from meridian.common.llm_client import LLMClient
from meridian.common.results import latency_summary, write_result
from meridian.eval.datasets import load_answer_set, load_out_of_scope_set

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/eval/e2.yaml"
REFUSAL_STATUS_SET = {"out_of_scope", "insufficient", "clarify"}
RAGAS_METRIC_LIST = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
)


class RagasRunConfig(BaseModel):
    """RAGAS execution limits."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: int = Field(gt=0)
    max_workers: int = Field(gt=0)
    max_retries: int = Field(ge=0)


class E2Config(VersionedConfig):
    """configs/eval/e2.yaml."""

    answer_set_path: str
    answer_split: str
    out_of_scope_path: str
    o_web_fallback: bool
    verify_sample_rate: float = Field(ge=0.0, le=1.0)
    ragas: RagasRunConfig
    output: str


# ============= Citation precision =============


def citation_pairs(answer: str, graded_list: list[dict]) -> list[tuple[str, str]]:
    """Return (sentence without markers, cited chunk text) for every corpus citation."""
    text_dict = {chunk["chunk_id"]: chunk["payload"]["text"] for chunk in graded_list}
    pair_list: list[tuple[str, str]] = []
    for sentence in split_sentences(answer):
        for kind, key in extract_citations(sentence):
            if kind == "c" and key in text_dict:
                pair_list.append((strip_citations(sentence), text_dict[key]))
    return pair_list


def judge_pair(llm: LLMClient, sentence: str, passage: str) -> bool:
    """Ask the judge whether one cited passage supports one sentence."""
    from meridian.agent.prompts import load_prompt

    try:
        reply = llm.chat_json(
            "judge",
            [
                {"role": "system", "content": load_prompt("verify")},
                {"role": "user", "content": f"Sentence: {sentence}\n\nPassage 1:\n{passage}"},
            ],
        )
    except ValueError as exc:
        logger.warning("judge reply unusable, counted unsupported", extra={"error": str(exc)})
        return False
    return reply.get("supported") is True


def citation_precision(llm: LLMClient, result_list: list) -> dict:
    """Return supported and total sentence-citation pairs over all answered items."""
    supported = total = 0
    for result in result_list:
        if result.status != "answered":
            continue
        for sentence, passage in citation_pairs(result.answer, result.graded):
            total += 1
            supported += int(judge_pair(llm, sentence, passage))
    return {
        "citation_precision": supported / total if total else None,
        "pairs_supported": supported,
        "pairs_total": total,
    }


def unsupported_sentence_rate(result_list: list) -> dict:
    """Return the share of draft sentences the verifier found unsupported."""
    unsupported = sum(len(result.verification.get("unsupported", [])) for result in result_list)
    sentences = sum(result.verification.get("n_sentences", 0) for result in result_list)
    return {
        "unsupported_sentence_rate": unsupported / sentences if sentences else None,
        "unsupported_sentences": unsupported,
        "draft_sentences": sentences,
    }


# ============= RAGAS =============


def ragas_samples(item_list: list[dict], result_list: list) -> list[dict]:
    """Return RAGAS samples from the agent's answers and contexts."""
    return [
        {
            "user_input": item["question"],
            "response": strip_citations(result.answer),
            "retrieved_contexts": result.contexts or [""],
            "reference": item["reference"],
        }
        for item, result in zip(item_list, result_list, strict=True)
    ]


def run_ragas(
    sample_list: list[dict], llm: LLMClient, metric_names: tuple[str, ...], run_config
) -> dict:
    """
    Score samples with RAGAS through the caching judge.

    Returns
    -------
    scores : dict
        Per metric: mean over scored samples and the scored count, so a mean over a partial set
        is never mistaken for a complete one.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from meridian.eval.ragas_bridge import CachedJudgeChatModel, DenseEmbeddings
    from meridian.ingest.embed import get_dense_encoder

    metric_dict = {
        "faithfulness": Faithfulness(),
        # strictness 1: Groq rejects n > 1, which RAGAS uses for strictness
        "answer_relevancy": ResponseRelevancy(strictness=1),
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }
    result = evaluate(
        dataset=EvaluationDataset.from_list(sample_list),
        metrics=[metric_dict[name] for name in metric_names],
        llm=CachedJudgeChatModel(llm=llm),
        embeddings=DenseEmbeddings(get_dense_encoder(get_index_config().dense)),
        run_config=RunConfig(
            timeout=run_config.timeout_s,
            max_workers=run_config.max_workers,
            max_retries=run_config.max_retries,
        ),
        show_progress=False,
    )
    frame = result.to_pandas()  # type: ignore[union-attr]
    score_dict: dict[str, dict] = {}
    for name in metric_names:
        column_values = list(frame[name]) if name in frame.columns else []
        value_list = [float(value) for value in column_values if not math.isnan(value)]
        score_dict[name] = {
            "mean": sum(value_list) / len(value_list) if value_list else None,
            "scored": len(value_list),
            "total": len(sample_list),
        }
    return score_dict


# ============= Runs =============


def answer_items(agent, item_list: list[dict]) -> list:
    """Answer each item in a fresh session."""
    return [agent.ask(item["question"], f"eval-{uuid.uuid4().hex}") for item in item_list]


def faithfulness_on_collection(collection: str, index_config: IndexConfig) -> float:
    """
    Return mean A-full faithfulness of the agent answering from one collection; used by the
    rebuild gate.

    Raises
    ------
    RuntimeError
        If no sample could be scored.
    """
    from meridian.agent.build import build_agent
    from meridian.common.llm_client import build_llm_client
    from meridian.retrieval.factory import build_retriever

    config = load_config(E2Config, DEFAULT_CONFIG_PATH)
    llm = build_llm_client()
    agent = build_agent(
        o_web_fallback=False,
        verify_sample_rate=config.verify_sample_rate,
        retriever=build_retriever(index_config, collection=collection),
        llm=llm,
    )
    item_list = load_answer_set(config.answer_set_path, "full")
    require_references(item_list)
    result_list = answer_items(agent, item_list)
    score = run_ragas(ragas_samples(item_list, result_list), llm, ("faithfulness",), config.ragas)
    mean = score["faithfulness"]["mean"]
    if mean is None:
        raise RuntimeError(f"faithfulness could not be scored on {collection}")
    return mean


def require_references(item_list: list[dict]) -> None:
    """Fail when any item lacks a reference answer, which the reference-based metrics need."""
    missing_list = [item["aid"] for item in item_list if not item.get("reference")]
    if missing_list:
        raise RuntimeError(
            f"{len(missing_list)} answer items lack references; "
            "run scripts/build_answer_references.py"
        )


def run_e2(config: E2Config) -> dict:
    """Run A and O through the agent and compute every E2 metric."""
    from meridian.agent.build import build_agent
    from meridian.common.llm_client import build_llm_client, usage_scope
    from meridian.ingest.index import alias_target, get_qdrant_client

    llm = build_llm_client()
    agent = build_agent(
        o_web_fallback=config.o_web_fallback, verify_sample_rate=config.verify_sample_rate, llm=llm
    )
    item_list = load_answer_set(config.answer_set_path, config.answer_split)
    require_references(item_list)
    out_of_scope_list = load_out_of_scope_set(config.out_of_scope_path)

    with usage_scope() as answer_meter:
        result_list = answer_items(agent, item_list)
    with usage_scope() as judge_meter:
        ragas_scores = run_ragas(
            ragas_samples(item_list, result_list), llm, RAGAS_METRIC_LIST, config.ragas
        )
        precision = citation_precision(llm, result_list)
    oos_result_list = answer_items(agent, out_of_scope_list)

    refused = sum(1 for result in oos_result_list if result.status in REFUSAL_STATUS_SET)
    status_count: dict[str, int] = {}
    for result in result_list:
        status_count[result.status] = status_count.get(result.status, 0) + 1
    models_config = get_models_config()
    return {
        "index": {"collection": alias_target(get_qdrant_client(), get_index_config().alias)},
        "models": {role: models_config.role(role).model for role in models_config.roles},
        "answer_set": {
            "path": config.answer_set_path,
            "split": config.answer_split,
            "n": len(item_list),
        },
        "ragas": ragas_scores,
        **precision,
        **unsupported_sentence_rate(result_list),
        "answer_status": status_count,
        "refusal": {
            "n": len(out_of_scope_list),
            "refused": refused,
            "refusal_accuracy": refused / len(out_of_scope_list) if out_of_scope_list else None,
            "by_status": {
                status: sum(1 for result in oos_result_list if result.status == status)
                for status in {result.status for result in oos_result_list}
            },
        },
        "latency": latency_summary([result.latency_s for result in result_list]),
        "llm_calls": {
            "answering": answer_meter.calls_by_role(),
            "judging": judge_meter.calls_by_role(),
        },
    }


def cli_eval(args) -> int:
    """`meridian eval answers`: run E2 and write its results file."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(E2Config, config_path)
    metrics = run_e2(config)
    output_path = write_result(
        config.output,
        metrics,
        config_path,
        notes=(
            "RAGAS scores use the contexts the agent actually used; means are over scored "
            "samples, with scored counts beside them. Web fallback is disabled so every context "
            "is a corpus chunk. The judge role is a different model from the generator."
        ),
    )
    print(Path(output_path).read_text()[:2000])
    return 0
