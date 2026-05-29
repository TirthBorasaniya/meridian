"""BGE embedder with the BAAI asymmetric instruction prefix.

BAAI/bge-small-en-v1.5 requires the query instruction prefix on search queries
but no prefix on documents. Encoding a query without the prefix yields
embeddings from a different distribution than the documents, depressing
similarity scores and degrading retrieval. This module enforces the asymmetry:
queries are always prefixed, documents are never prefixed.
"""

from functools import lru_cache

from meridian.config import get_settings


class BGEEmbedder:
    """Wrapper around a SentenceTransformer enforcing BGE prefix asymmetry."""

    def __init__(self, model_name: str, query_prefix: str) -> None:
        # Imported lazily so that importing this module does not load torch and
        # sentence-transformers until an embedder is actually constructed.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the asymmetric instruction prefix.

        Parameters
        ----------
        query : str
            The raw user query, without any prefix.

        Returns
        -------
        list of float
            The L2-normalised query embedding.
        """
        prefixed_query = f"{self._query_prefix}{query}"
        vector = self._model.encode(prefixed_query, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, text_list: list[str]) -> list[list[float]]:
        """Embed document chunks with no prefix.

        Parameters
        ----------
        text_list : list of str
            Document chunk texts.

        Returns
        -------
        list of list of float
            One L2-normalised embedding per input chunk, in input order.
        """
        vectors = self._model.encode(
            text_list, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> BGEEmbedder:
    """Return the process-wide cached BGE embedder."""
    settings = get_settings()
    return BGEEmbedder(settings.embedding_model, settings.bge_query_prefix)
