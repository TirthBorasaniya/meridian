"""Entrypoint for scoring sampled production traces with RAGAS.

Fetches the most recent completed traces from Langfuse, extracts the query,
retrieved context, and generated answer captured by the ``@observe``-wrapped
``retrieve``, ``grade_documents``, and ``generate`` node spans, and scores
faithfulness and answer relevancy against what the system actually returned
in production. This complements the offline 75-question suite in
``scripts/evaluate.py``, which runs against a fixed test set rather than live
traffic.

This script is not required for the main application to run. If Langfuse
credentials are not configured, it logs a warning and exits without error.

Examples
--------
Score the last 50 completed traces::

    python scripts/score_production_traces.py --n 50
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("score_production_traces")

RESULTS_PATH = os.path.join("data", "eval_results", "production_scores.json")


def _extract_sample(trace) -> dict | None:
    """Extract a RAGAS-shaped sample from a single Langfuse trace.

    Parameters
    ----------
    trace : object
        A trace object returned by the Langfuse SDK's ``fetch_traces``.

    Returns
    -------
    dict or None
        A dict with ``user_input``, ``retrieved_contexts``, and ``response``
        keys, or ``None`` if the trace does not contain a completed query.

    Notes
    -----
    Node functions are wrapped with the Langfuse ``@observe`` decorator (see
    ``meridian.graph.nodes``), which captures each function's arguments and
    return value as the span's input and output. This function reads the
    ``generate`` and ``grade_documents`` observations off the trace to
    recover the context and answer the graph actually used. Verify the
    Langfuse SDK's trace and observation attribute names at run time; they
    have moved across SDK versions.
    """
    observation_list = getattr(trace, "observations", None) or []
    generate_observation = next(
        (o for o in observation_list if getattr(o, "name", "") == "generate"), None
    )
    grade_observation = next(
        (o for o in observation_list if getattr(o, "name", "") == "grade_documents"), None
    )
    if generate_observation is None:
        return None

    generation_output = getattr(generate_observation, "output", None) or {}
    answer = generation_output.get("generation", "") if isinstance(generation_output, dict) else ""
    if not answer:
        return None

    context_list: list[str] = []
    if grade_observation is not None:
        grade_output = getattr(grade_observation, "output", None) or {}
        graded_docs = grade_output.get("graded_docs", []) if isinstance(grade_output, dict) else []
        context_list = [doc.get("text", "") for doc in graded_docs]

    query = getattr(trace, "input", None)
    if isinstance(query, dict):
        query = query.get("query", "")
    query = query or ""

    return {
        "user_input": query,
        "retrieved_contexts": context_list or [""],
        "response": answer,
    }


def _fetch_samples(num_traces: int) -> list[dict]:
    """Fetch and extract RAGAS samples from the last ``num_traces`` traces."""
    from langfuse import Langfuse

    from meridian.config import get_settings

    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("Langfuse credentials are not configured; skipping production scoring.")
        return []

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    response = client.fetch_traces(limit=num_traces)
    trace_list = getattr(response, "data", response)
    sample_list = [_extract_sample(trace) for trace in trace_list]
    return [sample for sample in sample_list if sample is not None]


def _score_samples(sample_list: list[dict]) -> dict:
    """Run RAGAS faithfulness and answer relevancy over the extracted samples."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness

    from meridian.config import get_settings

    settings = get_settings()
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
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    dataframe = result.to_pandas()
    numeric_means = dataframe.select_dtypes(include="number").mean()
    return {metric: float(value) for metric, value in numeric_means.items()}


def main() -> None:
    """Parse arguments, fetch production traces, score them, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Score sampled production traces from Langfuse with RAGAS."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of most recent completed traces to sample. Defaults to 50.",
    )
    args = parser.parse_args()

    sample_list = _fetch_samples(args.n)
    if not sample_list:
        logger.warning("No scoreable traces found; exiting without writing results.")
        return

    score_dict = _score_samples(sample_list)

    print(f"Production RAGAS scores (n={len(sample_list)} traces):")
    for metric_name, value in score_dict.items():
        print(f"  {metric_name}: {value:.4f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump({"num_traces": len(sample_list), "scores": score_dict}, results_file, indent=2)
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
