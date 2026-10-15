"""Download the dense, sparse, and reranker models into the image at build time."""

from meridian.common.config import get_index_config
from meridian.ingest.embed import DenseEncoder, SparseEncoder
from meridian.retrieval.rerank import Reranker


def main() -> None:
    """Instantiate each model once so its weights land in the image cache."""
    index_config = get_index_config()
    DenseEncoder(index_config.dense)
    SparseEncoder(index_config.sparse)
    Reranker(index_config.reranker)
    print("models cached")


if __name__ == "__main__":
    main()
