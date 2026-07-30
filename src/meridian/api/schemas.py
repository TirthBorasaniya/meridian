"""Pydantic request and response models for the API."""

from pydantic import BaseModel, Field

from meridian.config import get_settings

# Resolved once at import. The cap bounds the prompt that reaches the paid
# generation model; identifier fields are bounded because they are used as
# checkpoint and session-memory keys.
MAX_QUERY_LENGTH = get_settings().max_query_length
MAX_IDENTIFIER_LENGTH = 128


class QueryRequest(BaseModel):
    """Body for the ``/query`` endpoint."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="The user's natural-language question.",
    )
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
        description="Checkpoint thread id; reuse to resume a conversation.",
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
        description=(
            "Cross-session memory key; reuse to recall prior turns across "
            "process restarts. Defaults to thread_id when not given."
        ),
    )


class RetrievedDocument(BaseModel):
    """A graded document surfaced alongside an answer."""

    chunk_id: str
    arxiv_id: str
    title: str
    score: float


class QueryResponse(BaseModel):
    """Response for the ``/query`` endpoint."""

    query: str
    answer: str
    source: str  # "corpus" | "web"
    query_type: str
    num_documents: int
    iteration_count: int
    documents: list[RetrievedDocument]


class HealthResponse(BaseModel):
    """Response for the ``/health`` endpoint."""

    status: str  # "ok" | "degraded"
    collection: str
    indexed_points: int


class EvalSummaryResponse(BaseModel):
    """Response for the ``/eval-summary`` endpoint."""

    available: bool
    timestamp: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
