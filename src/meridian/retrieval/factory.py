"""Construction of a retriever over the live alias with the configured models."""

from qdrant_client import QdrantClient

from meridian.common.config import IndexConfig, get_index_config
from meridian.ingest.embed import get_dense_encoder, get_sparse_encoder
from meridian.ingest.index import get_qdrant_client
from meridian.retrieval.hybrid import HybridRetriever
from meridian.retrieval.rerank import get_reranker


def build_retriever(
    index_config: IndexConfig | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
    o_with_reranker: bool = True,
) -> HybridRetriever:
    """
    Return a hybrid retriever with the configured dense, sparse, and reranker models.

    Parameters
    ----------
    index_config : IndexConfig or None
        Defaults to configs/index.yaml.
    client : QdrantClient or None
        Defaults to the process-wide client.
    collection : str or None
        Collection or alias to search; defaults to the live alias.
    o_with_reranker : bool
        Load the cross-encoder.

    Returns
    -------
    retriever : HybridRetriever
        Ready retriever.
    """
    config = index_config or get_index_config()
    return HybridRetriever(
        client or get_qdrant_client(),
        collection or config.alias,
        get_dense_encoder(config.dense),
        get_sparse_encoder(config.sparse),
        get_reranker(config.reranker) if o_with_reranker else None,
        config.retrieval,
    )
