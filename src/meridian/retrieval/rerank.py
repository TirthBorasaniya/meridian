"""Cross-encoder reranking of fused candidates.

Scores are passed through the model's default sigmoid activation, so they lie in [0, 1] and
can be compared against the agent's accept and reject score bands.
"""

import logging
from functools import lru_cache

from meridian.common.config import RerankerConfig

logger = logging.getLogger(__name__)


class Reranker:
    """Sentence Transformers cross-encoder wrapper."""

    def __init__(self, reranker_config: RerankerConfig, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = reranker_config.model
        self.batch_size = reranker_config.batch_size
        self._model = CrossEncoder(reranker_config.model, device=device)

    def score(self, query: str, text_list: list[str]) -> list[float]:
        """
        Score each text against the query.

        Parameters
        ----------
        query : str
            The search query.
        text_list : list of str
            Candidate passages.

        Returns
        -------
        score_list : list of float
            One score in [0, 1] per passage, in input order.
        """
        if not text_list:
            return []
        score_array = self._model.predict(
            [(query, text) for text in text_list],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in score_array]


@lru_cache(maxsize=2)
def get_reranker(reranker_config: RerankerConfig) -> Reranker:
    """Return a process-wide reranker per config."""
    logger.info("loading reranker", extra={"model": reranker_config.model})
    return Reranker(reranker_config)
