"""Adapters that let RAGAS use the project's caching LLM client (judge role) and the configured
dense encoder, so every judgment is cached by the SHA-256 key and runs at temperature 0.
"""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from meridian.common.llm_client import LLMClient

ROLE_BY_MESSAGE_TYPE = {"human": "user", "ai": "assistant", "system": "system"}


class CachedJudgeChatModel(BaseChatModel):
    """LangChain chat model that sends every request through `LLMClient.chat` as one role."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLMClient
    role: str = "judge"

    @property
    def _llm_type(self) -> str:
        return "meridian-cached-judge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_list = [
            {
                "role": ROLE_BY_MESSAGE_TYPE.get(message.type, "user"),
                "content": str(message.content),
            }
            for message in messages
        ]
        response = self.llm.chat(self.role, message_list)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.text))])


class DenseEmbeddings(Embeddings):
    """LangChain embeddings over the configured dense encoder (used by answer relevancy)."""

    def __init__(self, dense_encoder) -> None:
        self._encoder = dense_encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents without the query prefix."""
        return [vector.tolist() for vector in self._encoder.encode_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the query prefix."""
        return self._encoder.encode_queries([text])[0].tolist()
