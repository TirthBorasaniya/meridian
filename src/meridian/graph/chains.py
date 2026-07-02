"""Structured-output LLM chains backed by Groq.

Routing, hallucination grading, answer grading, and query rewriting run on the
8B model and return validated Pydantic objects via ``with_structured_output``.
Final answer generation runs on the 70B model and returns a string. Model
tiering keeps classification cheap while reserving the larger model for the one
call where reasoning capacity matters.
"""

from functools import lru_cache
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from meridian.config import get_settings


class RouteQuery(BaseModel):
    """Classification of a user query into a coarse type."""

    query_type: Literal["factual", "comparative", "methodological"] = Field(
        description="The category that best describes the query."
    )


class GradeHallucinations(BaseModel):
    """Binary judgement of whether a generation is grounded in the context."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer contains claims not supported by the context, else 'no'."
    )


class GradeAnswer(BaseModel):
    """Binary judgement of whether an answer resolves the query."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer addresses the question, else 'no'."
    )


class RewrittenQuery(BaseModel):
    """A reformulated query optimised for retrieval."""

    rewritten_query: str = Field(description="The improved, retrieval-optimised query.")


@lru_cache(maxsize=1)
def _grading_llm() -> ChatGroq:
    """Return the cached 8B grading model with rate-limit backoff."""
    settings = get_settings()
    return ChatGroq(
        model=settings.grading_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_retries=5,
    )


@lru_cache(maxsize=1)
def _generation_llm() -> ChatGroq:
    """Return the cached 70B generation model with rate-limit backoff."""
    settings = get_settings()
    return ChatGroq(
        model=settings.generation_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_retries=5,
    )


_ROUTER_SYSTEM = (
    "You are a query classifier for a retrieval system over LLM reasoning and "
    "evaluation papers. Classify the query as 'factual' (asks for a specific "
    "fact or definition), 'comparative' (asks to compare methods or results), "
    "or 'methodological' (asks how something is done or measured). "
    "Respond with a JSON object with exactly one key, 'query_type', whose value "
    "is one of 'factual', 'comparative', or 'methodological'."
)

_HALLUCINATION_SYSTEM = (
    "You are a grader assessing whether an answer is grounded in the provided "
    "context. Return 'yes' only if the answer asserts facts that are not "
    "supported by the context. Return 'no' if every claim is supported. "
    "Respond with a JSON object with exactly one key, 'binary_score', whose "
    "value is 'yes' or 'no'."
)

_ANSWER_SYSTEM = (
    "You are a grader assessing whether an answer resolves the user's question. "
    "Return 'yes' if the answer directly addresses the question, else 'no'. "
    "Respond with a JSON object with exactly one key, 'binary_score', whose "
    "value is 'yes' or 'no'."
)

_REWRITE_SYSTEM = (
    "You reformulate a user question into a single improved query that is more "
    "effective for dense and sparse retrieval. Preserve the original intent, "
    "expand key technical terms, and remove conversational filler. "
    "Respond with a JSON object with exactly one key, 'rewritten_query', whose "
    "value is the improved query string."
)

_GENERATION_SYSTEM = (
    "You are a precise technical assistant answering questions about LLM "
    "reasoning and evaluation research. Answer using only the provided context. "
    "Cite supporting passages by their bracketed index. If the context is "
    "insufficient to answer, state that explicitly rather than speculating. "
    "Use formal technical language. Prior conversation turns, if any, are "
    "provided for continuity only; ground the answer in the context, not in "
    "unverified claims from earlier turns."
)


@lru_cache(maxsize=1)
def get_router_chain() -> Runnable:
    """Return the query-routing chain producing a :class:`RouteQuery`."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", _ROUTER_SYSTEM), ("human", "Query: {query}")]
    )
    return prompt | _grading_llm().with_structured_output(RouteQuery, method="json_mode")


@lru_cache(maxsize=1)
def get_hallucination_chain() -> Runnable:
    """Return the hallucination-grading chain producing a :class:`GradeHallucinations`."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _HALLUCINATION_SYSTEM),
            ("human", "Context:\n{context}\n\nAnswer:\n{generation}"),
        ]
    )
    return prompt | _grading_llm().with_structured_output(GradeHallucinations, method="json_mode")


@lru_cache(maxsize=1)
def get_answer_chain() -> Runnable:
    """Return the answer-grading chain producing a :class:`GradeAnswer`."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _ANSWER_SYSTEM),
            ("human", "Question:\n{query}\n\nAnswer:\n{generation}"),
        ]
    )
    return prompt | _grading_llm().with_structured_output(GradeAnswer, method="json_mode")


@lru_cache(maxsize=1)
def get_rewrite_chain() -> Runnable:
    """Return the query-rewrite chain producing a :class:`RewrittenQuery`."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", _REWRITE_SYSTEM), ("human", "Original question: {query}")]
    )
    return prompt | _grading_llm().with_structured_output(RewrittenQuery, method="json_mode")


@lru_cache(maxsize=1)
def get_generation_chain() -> Runnable:
    """Return the answer-generation chain producing a string."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _GENERATION_SYSTEM),
            (
                "human",
                "Conversation history:\n{conversation_history}\n\n"
                "Question: {query}\n\nContext:\n{context}",
            ),
        ]
    )
    return prompt | _generation_llm() | StrOutputParser()
