"""Ranking metrics over span-labeled queries: recall@k, MRR@k, and nDCG@k.

A query has one or more labeled spans. Each retrieved chunk covers a set of those spans
(see `meridian.eval.spans`). A span counts once: recall@k is the share of spans covered by
the top k, and nDCG gives gain 1 to a rank only when it covers a span not covered earlier, so
near-duplicate chunks of the same span are not rewarded twice.
"""

import math

from meridian.eval.spans import LabeledSpan, covered_span_indices


def coverage_by_rank(payload_list: list[dict], span_list: list[LabeledSpan]) -> list[set[int]]:
    """Return, per rank, the indices of labeled spans the chunk at that rank covers."""
    return [covered_span_indices(payload, span_list) for payload in payload_list]


def recall_at_k(covered_list: list[set[int]], n_spans: int, k: int) -> float:
    """Share of labeled spans covered by at least one of the top `k` chunks."""
    if n_spans == 0:
        return 0.0
    covered_set: set[int] = set().union(*covered_list[:k]) if covered_list[:k] else set()
    return len(covered_set) / n_spans


def mrr_at_k(covered_list: list[set[int]], k: int) -> float:
    """Reciprocal of the first rank (1-based) within `k` that covers any span, else 0."""
    for rank, covered_set in enumerate(covered_list[:k], start=1):
        if covered_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(covered_list: list[set[int]], n_spans: int, k: int) -> float:
    """
    Binary-gain nDCG where a rank gains 1 only for covering a not yet covered span.

    Parameters
    ----------
    covered_list : list of set of int
        Covered span indices per rank, best first.
    n_spans : int
        Number of labeled spans of the query.
    k : int
        Cutoff.

    Returns
    -------
    ndcg : float
        In [0, 1].
    """
    if n_spans == 0:
        return 0.0
    seen_set: set[int] = set()
    dcg = 0.0
    for rank, covered_set in enumerate(covered_list[:k], start=1):
        new_set = covered_set - seen_set
        if new_set:
            dcg += 1.0 / math.log2(rank + 1)
            seen_set |= new_set
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_spans, k) + 1))
    return dcg / ideal


def query_metrics(payload_list: list[dict], span_list: list[LabeledSpan]) -> dict[str, float]:
    """Return recall@5, recall@10, MRR@10, and nDCG@10 for one query."""
    covered_list = coverage_by_rank(payload_list, span_list)
    n_spans = len(span_list)
    return {
        "recall_at_5": recall_at_k(covered_list, n_spans, 5),
        "recall_at_10": recall_at_k(covered_list, n_spans, 10),
        "mrr_at_10": mrr_at_k(covered_list, 10),
        "ndcg_at_10": ndcg_at_k(covered_list, n_spans, 10),
    }


def mean_metrics(per_query_list: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric over queries."""
    if not per_query_list:
        return {}
    return {
        name: sum(metrics[name] for metrics in per_query_list) / len(per_query_list)
        for name in per_query_list[0]
    }
