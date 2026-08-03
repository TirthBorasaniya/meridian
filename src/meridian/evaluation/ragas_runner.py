"""RAGAS evaluation against the chunks the retriever actually returned.

For each evaluation question the compiled graph is run end to end. The answer
and the exact contexts the graph used (graded corpus chunks, or the web result
when the CRAG fallback fired) are scored with RAGAS. Scoring against the
retrieved contexts, not the full corpus, is the only configuration that
measures what the system actually did.

Target metrics after tuning: faithfulness >= 0.87, context precision >= 0.81.
"""

import json
import os
from datetime import UTC, datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import SecretStr
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from meridian.config import get_settings
from meridian.evaluation.question_set import get_questions
from meridian.graph.chains import build_chat_groq
from meridian.graph.graph import run_query

EVAL_RESULTS_DIR = "data/eval_results"
LATEST_SUMMARY_PATH = os.path.join(EVAL_RESULTS_DIR, "latest.json")


def build_judge_llm():
    """Return the LangChain chat model RAGAS uses to grade samples.

    The judge is configured independently of the generation model. Grading a
    model's output with that same model makes faithfulness partly
    self-assessed, so a judge equal to the generation model is rejected. The
    OpenAI and Anthropic providers give a fully independent judge; the Groq
    provider keeps the run on one API key using a different model family.

    Returns
    -------
    BaseChatModel
        A chat model for ``LangchainLLMWrapper``.

    Raises
    ------
    ValueError
        If the configured provider is unknown, or its API key is not set.
    ImportError
        If the provider's LangChain integration package is not installed.
    """
    settings = get_settings()
    provider = settings.ragas_judge_provider.strip().lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("RAGAS_JUDGE_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.ragas_judge_model or "gpt-4o-mini",
            temperature=0.0,
            api_key=SecretStr(settings.openai_api_key),
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "RAGAS_JUDGE_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=settings.ragas_judge_model or "claude-sonnet-4-5",
            temperature=0.0,
            api_key=SecretStr(settings.anthropic_api_key),
            timeout=None,
            stop=None,
        )

    if provider == "groq":
        judge_model = settings.ragas_judge_model or settings.generation_model
        if judge_model == settings.generation_model:
            raise ValueError(
                "The RAGAS judge must differ from the generation model "
                f"({settings.generation_model}); grading a model's own output makes "
                "faithfulness self-assessed. Set RAGAS_JUDGE_MODEL to another model."
            )
        # The gpt-oss judges are reasoning models. Left unbounded they spend
        # the whole response budget on reasoning tokens, which surfaces as
        # LLMDidNotFinishException or a request timeout rather than a score.
        return build_chat_groq(judge_model, max_tokens=4096, reasoning_effort="low")

    raise ValueError(
        f"Unknown RAGAS_JUDGE_PROVIDER: {provider!r}. Expected 'openai', 'anthropic', or 'groq'."
    )


def _extract_contexts(state: dict) -> list[str]:
    """Return the contexts the graph actually used for an answer."""
    if state.get("source") == "web" and state.get("web_search_result"):
        return [state["web_search_result"]]
    return [doc.get("text", "") for doc in state.get("graded_docs", [])]


def _build_samples(question_list: list[dict], thread_prefix: str) -> list[dict]:
    """Run the graph over each question and assemble RAGAS samples."""
    sample_list: list[dict] = []
    for index, question_record in enumerate(question_list):
        question = question_record["question"]
        final_state = run_query(question, thread_id=f"{thread_prefix}-{index}")
        sample_list.append(
            {
                "user_input": question,
                "retrieved_contexts": _extract_contexts(final_state),
                "response": final_state.get("generation", ""),
                "reference": question_record.get("reference", "") or "",
            }
        )
    return sample_list


def _summarize(result) -> dict:
    """Reduce a RAGAS result to mean metric scores.

    Means skip NaN entries, which RAGAS writes when a judge call fails. The
    companion :func:`_score_counts` records how many samples actually scored
    so that a mean over a partial set is never mistaken for a complete one.
    """
    dataframe = result.to_pandas()
    numeric_means = dataframe.select_dtypes(include="number").mean()
    return {metric: float(value) for metric, value in numeric_means.items()}


def _score_counts(result, total: int) -> dict:
    """Return the number of samples that produced a score for each metric."""
    dataframe = result.to_pandas()
    numeric_frame = dataframe.select_dtypes(include="number")
    return {
        str(metric): {"scored": int(numeric_frame[metric].notna().sum()), "total": total}
        for metric in numeric_frame.columns
    }


def _write_results(
    summary_dict: dict,
    output_dir: str,
    judge_dict: dict | None = None,
    num_questions: int | None = None,
    counts_dict: dict | None = None,
) -> str:
    """Persist the summary to a timestamped file and update latest.json.

    The judge identity is recorded alongside the scores. A RAGAS number is not
    interpretable without knowing which model produced it, so the two are kept
    together in the artifact rather than only in the run logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload_dict = {
        "timestamp": timestamp,
        "num_questions": num_questions,
        "judge": judge_dict or {},
        "scores": summary_dict,
        "scored_counts": counts_dict or {},
    }

    timestamped_path = os.path.join(output_dir, f"ragas_{timestamp}.json")
    with open(timestamped_path, "w", encoding="utf-8") as result_file:
        json.dump(payload_dict, result_file, indent=2)
    with open(LATEST_SUMMARY_PATH, "w", encoding="utf-8") as latest_file:
        json.dump(payload_dict, latest_file, indent=2)
    return timestamped_path


def run_evaluation(
    num_questions: int | None = None,
    thread_prefix: str = "eval",
    output_dir: str = EVAL_RESULTS_DIR,
) -> dict:
    """Run the RAGAS suite and persist the metric summary.

    Parameters
    ----------
    num_questions : int or None, optional
        Number of questions to evaluate. Defaults to the full 75-question set.
        The sequencing guidance recommends a 20-question smoke run first.
    thread_prefix : str, optional
        Prefix for per-question checkpoint thread ids. Defaults to ``"eval"``.
    output_dir : str, optional
        Directory for result files. Defaults to ``data/eval_results``.

    Returns
    -------
    dict
        Mean metric scores keyed by metric name.
    """
    settings = get_settings()
    question_list = get_questions(num_questions)
    sample_list = _build_samples(question_list, thread_prefix)

    dataset = EvaluationDataset.from_list(sample_list)
    evaluator_llm = LangchainLLMWrapper(build_judge_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )
    metric_list = [
        Faithfulness(),
        LLMContextPrecisionWithoutReference(),
        # strictness=1 because RAGAS passes strictness through as the OpenAI
        # ``n`` parameter, and the Groq API rejects n > 1. The default of 3
        # makes every answer-relevancy job fail with a 400 rather than score.
        ResponseRelevancy(strictness=1),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metric_list,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        # Serialised with a generous timeout: the judge is rate limited per
        # minute, and parallel workers trigger 429s that surface as NaN scores.
        run_config=RunConfig(timeout=600, max_workers=4, max_retries=10),
    )
    summary_dict = _summarize(result)
    counts_dict = _score_counts(result, len(sample_list))
    _write_results(
        summary_dict,
        output_dir,
        counts_dict=counts_dict,
        judge_dict={
            "provider": settings.ragas_judge_provider,
            "model": settings.ragas_judge_model,
            "generation_model": settings.generation_model,
        },
        num_questions=len(sample_list),
    )
    return summary_dict


def load_latest_summary() -> dict | None:
    """Return the most recent persisted evaluation summary, or None."""
    if not os.path.exists(LATEST_SUMMARY_PATH):
        return None
    with open(LATEST_SUMMARY_PATH, encoding="utf-8") as summary_file:
        return json.load(summary_file)
