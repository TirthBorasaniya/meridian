"""Regression gates. The fast gate runs R-fast through the full retrieval pipeline after a daily
batch and fails when recall@5 drops more than `max_regression` below the stored baseline; the
caller then deletes the batch's points. The full gate (rebuilds) compares a candidate version
with the live one on R-full and A-full.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from meridian.common.config import GateConfig, file_sha256
from meridian.eval.datasets import SPLIT_FAST, load_retrieval_set
from meridian.eval.retrieval_eval import SearchFunction, evaluate_search

logger = logging.getLogger(__name__)

# ============= Constants =============

# float subtraction noise must not turn an exact 0.02 drop into a failure
COMPARISON_DIGITS = 10
GATE_K = 10


# ============= Decision rules =============


def regressed(value: float, baseline: float, max_regression: float) -> bool:
    """Return True when `value` is more than `max_regression` below `baseline`."""
    return round(baseline - value, COMPARISON_DIGITS) > max_regression


@dataclass
class GateResult:
    """Outcome of one gate evaluation."""

    metric: str
    value: float
    baseline: float | None
    max_regression: float
    o_pass: bool
    o_baseline_created: bool
    n_queries: int
    collection: str
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for batch records and reports."""
        return asdict(self)

    def report(self) -> str:
        """Return a short human-readable report, used in alerts and GitHub issues."""
        baseline_text = "none" if self.baseline is None else f"{self.baseline:.4f}"
        verdict = "PASS" if self.o_pass else "FAIL"
        return (
            f"{verdict}: {self.metric}={self.value:.4f} on {self.n_queries} R-fast queries "
            f"against baseline {baseline_text} (max regression {self.max_regression}) "
            f"on {self.collection}."
        )


# ============= Baseline file =============


def load_baseline(path: str | Path) -> dict | None:
    """Return the stored baseline, or None when it does not exist."""
    baseline_path = Path(path)
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text())


def save_baseline(path: str | Path, result: GateResult, retrieval_set_path: str) -> dict:
    """Write the baseline from a gate result and return it."""
    baseline_dict = {
        "metric": result.metric,
        "value": result.value,
        "collection": result.collection,
        "n_queries": result.n_queries,
        "retrieval_set_sha256": file_sha256(retrieval_set_path),
        "updated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    baseline_path = Path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline_dict, indent=2) + "\n")
    return baseline_dict


# ============= Fast gate =============


def run_fast_gate(
    search: SearchFunction,
    gate_config: GateConfig,
    collection: str,
    baseline_path: str | Path | None = None,
    retrieval_set_path: str | None = None,
) -> GateResult:
    """
    Evaluate R-fast and compare the configured metric with the baseline.

    Parameters
    ----------
    search : callable
        Full-pipeline search over the collection under test, returning at least 10 chunks.
    gate_config : GateConfig
        Metric, tolerance, and file paths.
    collection : str
        Name recorded in the result.
    baseline_path : str, Path, or None
        Override of `gate_config.baseline_path`.
    retrieval_set_path : str or None
        Override of `gate_config.retrieval_set_path`.

    Returns
    -------
    result : GateResult
        A missing baseline is created from this run and the gate passes, because there is
        nothing to regress against; `o_baseline_created` records that.
    """
    set_path = retrieval_set_path or gate_config.retrieval_set_path
    item_list = load_retrieval_set(set_path, SPLIT_FAST)
    if not item_list:
        raise RuntimeError(f"R-fast is empty in {set_path}")
    summary = evaluate_search(search, item_list)
    value = float(summary[gate_config.fast_metric])
    path = baseline_path or gate_config.baseline_path
    baseline_dict = load_baseline(path)
    if baseline_dict is not None and baseline_dict.get("metric") != gate_config.fast_metric:
        raise ValueError(
            f"baseline metric {baseline_dict.get('metric')} differs from {gate_config.fast_metric}"
        )
    baseline = float(baseline_dict["value"]) if baseline_dict is not None else None
    o_pass = baseline is None or not regressed(value, baseline, gate_config.max_regression)
    result = GateResult(
        metric=gate_config.fast_metric,
        value=value,
        baseline=baseline,
        max_regression=gate_config.max_regression,
        o_pass=o_pass,
        o_baseline_created=baseline is None,
        n_queries=len(item_list),
        collection=collection,
        summary={key: val for key, val in summary.items() if key != "latency"},
    )
    logger.info(
        "fast gate",
        extra={"verdict": "pass" if o_pass else "fail", "value": value, "baseline": baseline},
    )
    return result


# ============= Full gate =============


@dataclass
class FullGateResult:
    """Comparison of a candidate version with the live version."""

    candidate: str
    live: str | None
    retrieval: dict
    faithfulness: dict
    o_pass: bool
    reason_list: list[str]

    def to_dict(self) -> dict:
        """Serialize for reports."""
        return asdict(self)


def decide_promotion(
    candidate_metrics: dict,
    live_metrics: dict | None,
    candidate_faithfulness: float | None,
    live_faithfulness: float | None,
    gate_config: GateConfig,
) -> tuple[bool, list[str]]:
    """
    Apply SPEC S3.7: promote unless recall@5 or nDCG@10 is worse by more than
    `max_regression`, or faithfulness is worse by more than `faithfulness_max_regression`.

    Parameters
    ----------
    candidate_metrics, live_metrics : dict
        R-full means of the candidate and live versions; `live_metrics` is None when no
        version is live yet.
    candidate_faithfulness, live_faithfulness : float or None
        A-full RAGAS faithfulness; None when the answer evaluation was skipped.
    gate_config : GateConfig
        Tolerances and metric names.

    Returns
    -------
    decision : tuple of (bool, list of str)
        Whether to promote, and the failing reasons.
    """
    if live_metrics is None:
        return True, []
    reason_list: list[str] = []
    for metric in gate_config.full_metrics:
        if regressed(candidate_metrics[metric], live_metrics[metric], gate_config.max_regression):
            reason_list.append(
                f"{metric} {candidate_metrics[metric]:.4f} < live {live_metrics[metric]:.4f} "
                f"- {gate_config.max_regression}"
            )
    if candidate_faithfulness is None or live_faithfulness is None:
        reason_list.append("faithfulness was not measured on both versions")
    elif regressed(
        candidate_faithfulness, live_faithfulness, gate_config.faithfulness_max_regression
    ):
        reason_list.append(
            f"faithfulness {candidate_faithfulness:.4f} < live {live_faithfulness:.4f} "
            f"- {gate_config.faithfulness_max_regression}"
        )
    return not reason_list, reason_list
