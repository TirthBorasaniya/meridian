"""Deterministic stand-ins for the dense encoder, sparse encoder, and cross-encoder."""

import re
import zlib

import numpy as np
from qdrant_client import models

WORD_PATTERN = re.compile(r"[a-z0-9]+")
DENSE_DIM = 32


def words(text: str) -> list[str]:
    """Lowercase word tokens."""
    return WORD_PATTERN.findall(text.lower())


def bucket(word: str, size: int) -> int:
    """Stable hash bucket of a word."""
    return zlib.crc32(word.encode("utf-8")) % size


class FakeDenseEncoder:
    """Bag-of-words vectors hashed into a small dimension, L2-normalized."""

    model_name = "fake-dense"
    query_prefix = ""
    cache = None

    @property
    def dimension(self) -> int:
        return DENSE_DIM

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DENSE_DIM, dtype=np.float32)
        for word in words(text):
            vector[bucket(word, DENSE_DIM)] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        return vector / norm

    def encode_documents(self, text_list: list[str]) -> np.ndarray:
        return (
            np.stack([self._vector(text) for text in text_list])
            if text_list
            else np.zeros((0, DENSE_DIM), dtype=np.float32)
        )

    def encode_queries(self, query_list: list[str]) -> np.ndarray:
        return self.encode_documents(query_list)

    def encode_query(self, query: str) -> list[float]:
        return self._vector(query).tolist()


class FakeSparseEncoder:
    """Term-frequency sparse vectors over hashed word ids."""

    model_name = "fake-sparse"

    def _sparse(self, text: str) -> models.SparseVector:
        count_dict: dict[int, float] = {}
        for word in words(text):
            index = bucket(word, 100_000)
            count_dict[index] = count_dict.get(index, 0.0) + 1.0
        if not count_dict:
            count_dict = {0: 1.0}
        index_list = sorted(count_dict)
        return models.SparseVector(
            indices=index_list, values=[count_dict[index] for index in index_list]
        )

    def encode_documents(self, text_list: list[str], batch_size: int = 256) -> list:
        return [self._sparse(text) for text in text_list]

    def encode_query(self, query: str) -> models.SparseVector:
        return self._sparse(query)


class FakeReranker:
    """Scores by the share of query words present in the passage."""

    model_name = "fake-reranker"

    def score(self, query: str, text_list: list[str]) -> list[float]:
        query_set = set(words(query))
        if not query_set:
            return [0.0 for _ in text_list]
        return [len(query_set & set(words(text))) / len(query_set) for text in text_list]


class ScriptedTransport:
    """LLM transport whose reply is computed by a function of the request."""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.request_list: list[tuple[str, list[dict], dict]] = []

    def complete(self, model, message_list, param_dict, on_delta):
        from meridian.common.llm_client import Completion

        self.request_list.append((model, message_list, param_dict))
        text = self.responder(model, message_list, param_dict)
        if on_delta is not None:
            for word in text.split(" "):
                on_delta(word + " ")
        return Completion(text=text, input_tokens=100, output_tokens=20)
