"""The paper-level exclusion list: no paper referenced by R, A, or O contributes training text."""

from collections.abc import Iterable

from meridian.eval.datasets import evaluation_paper_ids


class LeakageError(Exception):
    """A training pair comes from an evaluation paper."""


def build_exclusion_list() -> set[str]:
    """Return the arXiv ids excluded from training; missing evaluation files raise."""
    return evaluation_paper_ids()


def assert_no_leakage(pair_list: Iterable[dict], excluded_set: set[str]) -> int:
    """
    Raise when any pair's source paper is excluded; return the number of pairs checked.

    Parameters
    ----------
    pair_list : iterable of dict
        Training pairs, each with `arxiv_id` of its positive chunk.
    excluded_set : set of str
        Evaluation paper ids.

    Returns
    -------
    count : int
        Pairs checked.
    """
    count = 0
    for pair in pair_list:
        count += 1
        if pair["arxiv_id"] in excluded_set:
            raise LeakageError(f"training pair from evaluation paper {pair['arxiv_id']}")
        for negative in pair.get("negatives", []):
            if negative.get("arxiv_id") in excluded_set:
                raise LeakageError(f"negative from evaluation paper {negative['arxiv_id']}")
    return count
