"""Render README tables from `results/*.json` between `<!-- results:<name>:start -->` and
`<!-- results:<name>:end -->` markers. A missing results file renders as "not yet measured",
never as a number. `--check` exits non-zero when the README differs from the rendering (CI).

Usage::

    uv run python scripts/render_results.py          # rewrite README.md
    uv run python scripts/render_results.py --check  # verify only
"""

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
RESULTS_DIR = REPO_ROOT / "results"
BLOCK_PATTERN = re.compile(
    r"(<!-- results:(?P<name>[a-z0-9_]+):start -->\n)(?P<body>.*?)(<!-- results:(?P=name):end -->)",
    re.DOTALL,
)
MISSING = "not yet measured"
EXPERIMENT_FILE_DICT = {
    "e1": "e1_retrieval.json",
    "e2": "e2_answers.json",
    "e3": "e3_serving.json",
    "e4": "e4_ingest.json",
    "e5": "e5_embedder.json",
    "e6": "e6_demo.json",
}
MAKE_TARGET_DICT = {
    "e1": "make eval-retrieval",
    "e2": "make eval-answers",
    "e3": "make eval-serving",
    "e4": "make eval-ingest",
    "e5": "make eval-embedder",
    "e6": "make eval-demo",
}


# ============= Formatting =============


def number(value, digits: int = 4) -> str:
    """Format a number, or the missing marker for None."""
    if value is None:
        return MISSING
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{value:.{digits}f}"


def millis(seconds) -> str:
    """Format seconds as milliseconds."""
    return MISSING if seconds is None else f"{seconds * 1000:.0f} ms"


def secs(seconds) -> str:
    """Format seconds."""
    return MISSING if seconds is None else f"{seconds:.1f} s"


def row(*cell_list) -> str:
    """Return one Markdown table row."""
    return "| " + " | ".join(str(cell) for cell in cell_list) + " |"


def dig(data: dict | None, *key_list, default=None):
    """Read a nested key, returning `default` when any level is missing."""
    current = data
    for key in key_list:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def provenance(result: dict) -> str:
    """Return the run line printed under each table."""
    return (
        f"\nRun `{result['run_id']}`, commit `{result['git_sha']}`, "
        f"config `{result['config_path']}` (sha256 `{result['config_sha256'][:12]}`), "
        f"{result['environment']['hardware']}.\n"
    )


def missing_block(name: str) -> str:
    """Return the text of a block whose results file does not exist."""
    return (
        f"Not yet measured. Run `{MAKE_TARGET_DICT[name]}` to produce "
        f"`results/{EXPERIMENT_FILE_DICT[name]}`.\n"
    )


# ============= Renderers =============


def render_headline(results: dict) -> str:
    """Return the headline table."""
    e1 = dig(results, "e1", "metrics")
    e2 = dig(results, "e2", "metrics")
    e3 = dig(results, "e3", "metrics")
    e4 = dig(results, "e4", "metrics")
    e5 = dig(results, "e5", "metrics")
    e6 = dig(results, "e6", "metrics")
    faithfulness = dig(e2, "ragas", "faithfulness")
    faithfulness_text = (
        f"{number(faithfulness['mean'])} ({faithfulness['scored']}/{faithfulness['total']} scored)"
        if faithfulness and faithfulness.get("mean") is not None
        else MISSING
    )
    row_list = [
        (
            "Retrieval recall@5, hybrid plus rerank (R-full)",
            number(dig(e1, "configurations", "hybrid_rerank", "recall_at_5")),
            "E1",
        ),
        (
            "Retrieval nDCG@10, hybrid plus rerank (R-full)",
            number(dig(e1, "configurations", "hybrid_rerank", "ndcg_at_10")),
            "E1",
        ),
        ("RAGAS faithfulness (A)", faithfulness_text, "E2"),
        ("Citation precision (A)", number(dig(e2, "citation_precision")), "E2"),
        ("Refusal accuracy (O)", number(dig(e2, "refusal", "refusal_accuracy")), "E2"),
        ("End-to-end latency p95", secs(dig(e3, "score_band_grading", "latency", "p95_s")), "E3"),
        (
            "Cost per answer (USD)",
            number(dig(e3, "score_band_grading", "cost_usd_per_answer"), 5),
            "E3",
        ),
        ("Answer cache hit rate on repeats", number(dig(e3, "answer_cache", "hit_rate")), "E3"),
        (
            "Scheduled index runs recorded",
            number(dig(e4, "scheduled_runs", "summary", "runs")),
            "E4",
        ),
        (
            "Adapted embedder recall@5 change, full pipeline",
            number(dig(e5, "delta_adapted_minus_base", "hybrid_rerank", "recall_at_5")),
            "E5",
        ),
        ("Demo cold start", secs(dig(e6, "cold_start", "seconds")), "E6"),
    ]
    line_list = ["| Measure | Value | Source |", "|---|---|---|"]
    line_list += [f"| {label} | {value} | {source} |" for label, value, source in row_list]
    return "\n".join(line_list) + "\n"


def render_e1(result: dict) -> str:
    """Return the E1 tables."""
    metrics = result["metrics"]
    line_list = [
        f"R-{metrics['dataset']['split']}: {metrics['dataset']['n_items']} queries "
        f"({metrics['dataset']['n_invalid_excluded']} reviewed invalid and excluded); index "
        f"`{metrics['index']['collection']}`, {metrics['index']['papers']} papers, "
        f"{metrics['index']['points']} chunks.",
        "",
        "| Configuration | recall@5 | recall@10 | MRR@10 | nDCG@10 | p50 total | p95 total |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, summary in metrics["configurations"].items():
        latency = summary["latency"]["total"]
        line_list.append(
            f"| {name} | {number(summary['recall_at_5'])} | {number(summary['recall_at_10'])} | "
            f"{number(summary['mrr_at_10'])} | {number(summary['ndcg_at_10'])} | "
            f"{millis(latency['p50_s'])} | {millis(latency['p95_s'])} |"
        )
    stage_dict = metrics["configurations"].get("hybrid_rerank", {}).get("latency", {})
    if stage_dict:
        line_list += [
            "",
            "Stage latency, hybrid plus rerank:",
            "",
            "| Stage | p50 | p95 |",
            "|---|---|---|",
        ]
        for stage, summary in stage_dict.items():
            line_list.append(
                f"| {stage} | {millis(summary['p50_s'])} | {millis(summary['p95_s'])} |"
            )
    line_list += [
        "",
        "Prefetch limit sensitivity (hybrid plus rerank):",
        "",
        "| Prefetch limit | recall@5 | nDCG@10 | p95 total |",
        "|---|---|---|---|",
    ]
    for limit, summary in metrics["prefetch_sensitivity"]["by_limit"].items():
        line_list.append(
            f"| {limit} | {number(summary['recall_at_5'])} | {number(summary['ndcg_at_10'])} | "
            f"{millis(summary['latency']['total']['p95_s'])} |"
        )
    line_list += [
        "",
        "Chunk size sensitivity (hybrid plus rerank):",
        "",
        "| Chunk tokens | Chunks | recall@5 | nDCG@10 |",
        "|---|---|---|---|",
    ]
    for size, summary in metrics["chunk_size_sensitivity"]["by_chunk_tokens"].items():
        chunks = summary["build"].get("chunks", metrics["index"]["points"])
        line_list.append(
            row(size, chunks, number(summary["recall_at_5"]), number(summary["ndcg_at_10"]))
        )
    return "\n".join(line_list) + "\n" + provenance(result)


def render_e2(result: dict) -> str:
    """Return the E2 table."""
    metrics = result["metrics"]
    refusal = metrics["refusal"]
    line_list = ["| Metric | Value | Scored |", "|---|---|---|"]
    for name, score in metrics["ragas"].items():
        line_list.append(
            row(f"RAGAS {name}", number(score["mean"]), f"{score['scored']}/{score['total']}")
        )
    line_list += [
        row(
            "Citation precision",
            number(metrics["citation_precision"]),
            f"{metrics['pairs_supported']}/{metrics['pairs_total']} pairs",
        ),
        row(
            "Unsupported draft sentences",
            number(metrics["unsupported_sentence_rate"]),
            f"{metrics['unsupported_sentences']}/{metrics['draft_sentences']}",
        ),
        row(
            "Refusal accuracy on O",
            number(refusal["refusal_accuracy"]),
            f"{refusal['refused']}/{refusal['n']}",
        ),
        "",
        "Models: "
        + ", ".join(f"{role} `{model}`" for role, model in metrics["models"].items())
        + ".",
    ]
    return "\n".join(line_list) + "\n" + provenance(result)


def render_e3(result: dict) -> str:
    """Return the E3 table."""
    metrics = result["metrics"]
    band = metrics["score_band_grading"]
    llm_only = metrics["llm_grading_only"]
    semantic = metrics["semantic_cache"]
    distance = semantic["distance_threshold"]
    line_list = [
        "| Measure | Score-band grading | LLM grading only |",
        "|---|---|---|",
        row("End-to-end p50", secs(band["latency"]["p50_s"]), secs(llm_only["latency"]["p50_s"])),
        row("End-to-end p95", secs(band["latency"]["p95_s"]), secs(llm_only["latency"]["p95_s"])),
        row(
            "LLM calls per answer",
            number(band["llm_calls_per_answer"], 2),
            number(llm_only["llm_calls_per_answer"], 2),
        ),
        row(
            "Tokens per answer",
            number(band["tokens_per_answer"], 0),
            number(llm_only["tokens_per_answer"], 0),
        ),
        row(
            "Cost per answer (USD)",
            number(band["cost_usd_per_answer"], 5),
            number(llm_only["cost_usd_per_answer"], 5),
        ),
        "",
        "| Cache | Measure | Value |",
        "|---|---|---|",
        row(
            "Answer cache",
            "hit rate on exact and normalized repeats",
            number(metrics["answer_cache"]["hit_rate"]),
        ),
        row(
            "Answer cache",
            "hit latency p50",
            millis(metrics["answer_cache"]["hit_latency"]["p50_s"]),
        ),
        row("Embedding cache", "hit rate", number(metrics["embedding_cache"]["hit_rate"])),
        row(
            f"Semantic cache (distance {distance})",
            "same-intent hit rate",
            number(semantic["same_intent_hit_rate"]),
        ),
        row(
            f"Semantic cache (distance {distance})",
            "wrong-hit rate",
            number(semantic["wrong_hit_rate"]),
        ),
        "",
        f"Prices checked on: {metrics['prices']['checked_on']}.",
    ]
    return "\n".join(line_list) + "\n" + provenance(result)


def render_e4(result: dict) -> str:
    """Return the E4 tables."""
    metrics = result["metrics"]
    simulation = metrics["simulation"]
    line_list = [
        f"Simulated replay from {simulation['start_date']}, {simulation['days']} days, at most "
        f"{simulation['max_papers_per_day']} papers per day.",
        "",
        "| Day | Fetched | Skipped | Failed | Indexed | Chunks | Decision | recall@5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    recall_by_day = {
        point["day"]: point.get("recall_at_5") for point in simulation["recall_at_5_trajectory"]
    }
    for batch in simulation["batches"]:
        line_list.append(
            f"| {batch['day']} | {batch['papers_fetched']} | {batch['papers_skipped']} | "
            f"{batch['papers_failed']} | {batch['papers_indexed']} | {batch['chunks_added']} | "
            f"{batch['decision']} | {number(recall_by_day.get(batch['day']))} |"
        )
    summary = metrics["scheduled_runs"]["summary"]
    line_list += [
        "",
        f"Scheduled CI runs recorded: {summary['runs']}; "
        f"published snapshots: {summary['published']}; "
        f"by decision: {json.dumps(summary['by_decision'], sort_keys=True)}.",
    ]
    return "\n".join(line_list) + "\n" + provenance(result)


def render_e5(result: dict) -> str:
    """Return the E5 table."""
    metrics = result["metrics"]
    line_list = [
        "| Pipeline | Model | recall@5 | recall@10 | MRR@10 | nDCG@10 |",
        "|---|---|---|---|---|---|",
    ]
    for pipeline, model_dict in metrics["comparison"].items():
        for model_name, summary in model_dict.items():
            line_list.append(
                f"| {pipeline} | {model_name} | {number(summary['recall_at_5'])} | "
                f"{number(summary['recall_at_10'])} | {number(summary['mrr_at_10'])} | "
                f"{number(summary['ndcg_at_10'])} |"
            )
    training = metrics.get("training") or {}
    gate = metrics.get("gate_decision")
    line_list += [
        "",
        f"Training: {training.get('hardware', MISSING)}, {secs(training.get('wall_clock_s'))}.",
        "Gate decision: "
        + (
            MISSING
            if gate is None
            else ("promoted" if gate["promoted"] else "declined: " + "; ".join(gate["reasons"]))
        )
        + ".",
    ]
    return "\n".join(line_list) + "\n" + provenance(result)


def render_e6(result: dict) -> str:
    """Return the E6 table."""
    metrics = result["metrics"]
    rate = metrics["rate_limit"]
    burst = (
        f"{rate['allowed']} allowed, {rate['limited']} limited "
        f"(Retry-After {rate['retry_after_s']} s)"
    )
    line_list = [
        "| Measure | Value |",
        "|---|---|",
        row("Cold start (restart to ready)", secs(dig(metrics, "cold_start", "seconds"))),
        row(
            f"Latency p50, {metrics['questions']} scripted questions",
            secs(metrics["latency"]["p50_s"]),
        ),
        row("Latency p95", secs(metrics["latency"]["p95_s"])),
        row("Time to first token p50", secs(metrics["first_token"]["p50_s"])),
        row(f"Burst of {rate['burst']} searches", burst),
    ]
    return "\n".join(line_list) + "\n" + provenance(result)


RENDERER_DICT: dict[str, Callable[[dict], str]] = {
    "e1": render_e1,
    "e2": render_e2,
    "e3": render_e3,
    "e4": render_e4,
    "e5": render_e5,
    "e6": render_e6,
}


# ============= README =============


def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """Return the parsed results files that exist, keyed by experiment."""
    loaded: dict[str, dict] = {}
    for name, file_name in EXPERIMENT_FILE_DICT.items():
        path = results_dir / file_name
        if path.exists():
            loaded[name] = json.loads(path.read_text())
    return loaded


def render_block(name: str, results: dict[str, dict]) -> str:
    """Return the body of one marker block."""
    if name == "headline":
        return render_headline(results)
    if name not in RENDERER_DICT:
        raise KeyError(f"unknown results block {name!r}")
    if name not in results:
        return missing_block(name)
    return RENDERER_DICT[name](results[name])


def render_readme(text: str, results: dict[str, dict]) -> str:
    """Replace every marker block in the README text."""

    def replace(match: re.Match) -> str:
        return match.group(1) + render_block(match.group("name"), results) + match.group(4)

    return BLOCK_PATTERN.sub(replace, text)


def main() -> int:
    """Rewrite the README, or verify it with `--check`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = README_PATH.read_text()
    rendered = render_readme(current, load_results())
    if args.check:
        if rendered != current:
            print("README.md tables are stale; run `make results`", file=sys.stderr)
            return 1
        return 0
    README_PATH.write_text(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
