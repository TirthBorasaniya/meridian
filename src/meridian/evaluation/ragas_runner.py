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
from datetime import datetime, timezone

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

from meridian.config import get_settings
from meridian.evaluation.question_set import get_questions
from meridian.graph.graph import run_query

EVAL_RESULTS_DIR = "data/eval_results"
LATEST_SUMMARY_PATH = os.path.join(EVAL_RESULTS_DIR, "latest.json")


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
    """Reduce a RAGAS result to mean metric scores."""
    dataframe = result.to_pandas()
    numeric_means = dataframe.select_dtypes(include="number").mean()
    return {metric: float(value) for metric, value in numeric_means.items()}


def _write_results(summary_dict: dict, output_dir: str) -> str:
    """Persist the summary to a timestamped file and update latest.json."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload_dict = {"timestamp": timestamp, "scores": summary_dict}

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
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model=settings.generation_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
            max_retries=5,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )
    metric_list = [Faithfulness(), LLMContextPrecisionWithoutReference()]

    result = evaluate(
        dataset=dataset,
        metrics=metric_list,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    summary_dict = _summarize(result)
    _write_results(summary_dict, output_dir)
    return summary_dict


def load_latest_summary() -> dict | None:
    """Return the most recent persisted evaluation summary, or None."""
    if not os.path.exists(LATEST_SUMMARY_PATH):
        return None
    with open(LATEST_SUMMARY_PATH, encoding="utf-8") as summary_file:
        return json.load(summary_file)
