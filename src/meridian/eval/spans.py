"""Span-based relevance: a retrieved chunk is relevant to a labeled span when it is from the same
paper version and covers at least half of the span's characters (SPEC S6.1).
"""

from dataclasses import dataclass

# ============= Constants =============

COVERAGE_THRESHOLD = 0.5


@dataclass(frozen=True)
class LabeledSpan:
    """A relevant character span of one parsed paper version."""

    arxiv_id: str
    version: int
    start: int
    end: int

    @classmethod
    def from_dict(cls, span_dict: dict) -> "LabeledSpan":
        """Build from a dataset record entry."""
        return cls(
            arxiv_id=span_dict["arxiv_id"],
            version=int(span_dict["version"]),
            start=int(span_dict["start"]),
            end=int(span_dict["end"]),
        )

    def to_dict(self) -> dict:
        """Serialize for a dataset record."""
        return {
            "arxiv_id": self.arxiv_id,
            "version": self.version,
            "start": self.start,
            "end": self.end,
        }


def span_coverage(payload: dict, span: LabeledSpan) -> float:
    """
    Return the fraction of a labeled span covered by a chunk.

    Parameters
    ----------
    payload : dict
        Chunk payload with `arxiv_id`, `version`, `char_start`, `char_end`.
    span : LabeledSpan
        The labeled span.

    Returns
    -------
    coverage : float
        Overlap length divided by span length, in [0, 1]; 0 for another paper or version.
    """
    if payload.get("arxiv_id") != span.arxiv_id or int(payload.get("version", -1)) != span.version:
        return 0.0
    span_length = span.end - span.start
    if span_length <= 0:
        return 0.0
    overlap = min(int(payload["char_end"]), span.end) - max(int(payload["char_start"]), span.start)
    return max(0, overlap) / span_length


def covered_span_indices(
    payload: dict, span_list: list[LabeledSpan], threshold: float = COVERAGE_THRESHOLD
) -> set[int]:
    """Return the indices of the labeled spans that a chunk covers at or above `threshold`."""
    return {
        index for index, span in enumerate(span_list) if span_coverage(payload, span) >= threshold
    }
