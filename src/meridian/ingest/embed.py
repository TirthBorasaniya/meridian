"""Dense and sparse encoders for indexing and querying.

The dense side is the configured Sentence Transformers model with an asymmetric query prefix;
the sparse side is `Qdrant/bm25` through FastEmbed, whose IDF weighting is applied by Qdrant
through the collection's IDF modifier.
"""

import hashlib
import logging
from functools import lru_cache
from typing import Protocol

import numpy as np
from qdrant_client import models

from meridian.common.config import DenseConfig, SparseConfig

logger = logging.getLogger(__name__)


# ============= Cache protocol =============


class VectorCache(Protocol):
    """Key-value store for query embeddings."""

    def get_vector(self, key: str) -> list[float] | None: ...

    def set_vector(self, key: str, vector: list[float]) -> None: ...


def embedding_cache_key(model_name: str, text: str) -> str:
    """Return the embedding cache key: model name plus the SHA-256 of the text."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"emb:{model_name}:{text_hash}"


# ============= Dense =============


class DenseEncoder:
    """Sentence Transformers encoder with separate query and document prefixes."""

    def __init__(self, dense_config: DenseConfig, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = dense_config.model
        self.query_prefix = dense_config.query_prefix
        self.document_prefix = dense_config.document_prefix
        self.batch_size = dense_config.batch_size
        self._model = SentenceTransformer(dense_config.model, device=device)
        self.cache: VectorCache | None = None

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"{self.model_name} does not report an embedding dimension")
        return int(dimension)

    def encode_documents(self, text_list: list[str]) -> np.ndarray:
        """
        Embed document texts.

        Parameters
        ----------
        text_list : list of str
            Chunk texts.

        Returns
        -------
        matrix : np.ndarray
            L2-normalized embeddings, shape (len(text_list), dimension).
        """
        if not text_list:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._model.encode(
            [f"{self.document_prefix}{text}" for text in text_list],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_queries(self, query_list: list[str]) -> np.ndarray:
        """Embed queries with the query prefix; shape (len(query_list), dimension)."""
        return self._model.encode(
            [f"{self.query_prefix}{query}" for query in query_list],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> list[float]:
        """Embed one query, through the vector cache when one is attached."""
        cache_key = embedding_cache_key(self.model_name, f"{self.query_prefix}{query}")
        if self.cache is not None:
            cached_vector = self.cache.get_vector(cache_key)
            if cached_vector is not None:
                return cached_vector
        vector = self.encode_queries([query])[0].tolist()
        if self.cache is not None:
            self.cache.set_vector(cache_key, vector)
        return vector


# ============= Sparse =============


def to_sparse_vector(embedding) -> models.SparseVector:
    """Convert a FastEmbed sparse embedding to a Qdrant sparse vector."""
    return models.SparseVector(
        indices=[int(index) for index in embedding.indices],
        values=[float(value) for value in embedding.values],
    )


class SparseEncoder:
    """FastEmbed `Qdrant/bm25` encoder; term weights only, IDF is applied by Qdrant."""

    def __init__(self, sparse_config: SparseConfig, cache_dir: str | None = None) -> None:
        from fastembed import SparseTextEmbedding

        self.model_name = sparse_config.model
        self._model = SparseTextEmbedding(model_name=sparse_config.model, cache_dir=cache_dir)

    def encode_documents(self, text_list: list[str], batch_size: int = 256) -> list:
        """Return one Qdrant sparse vector per document."""
        return [
            to_sparse_vector(embedding)
            for embedding in self._model.embed(text_list, batch_size=batch_size)
        ]

    def encode_query(self, query: str) -> models.SparseVector:
        """Return the sparse query vector."""
        return to_sparse_vector(next(iter(self._model.query_embed(query))))


# ============= Cached constructors =============


@lru_cache(maxsize=4)
def get_dense_encoder(dense_config: DenseConfig) -> DenseEncoder:
    """Return a process-wide dense encoder per config."""
    logger.info("loading dense encoder", extra={"model": dense_config.model})
    return DenseEncoder(dense_config)


@lru_cache(maxsize=2)
def get_sparse_encoder(sparse_config: SparseConfig) -> SparseEncoder:
    """Return a process-wide sparse encoder per config."""
    logger.info("loading sparse encoder", extra={"model": sparse_config.model})
    return SparseEncoder(sparse_config)
