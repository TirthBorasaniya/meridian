"""Pydantic request and response models for the API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for the ``/query`` endpoint."""

    query: str = Field(description="The user's natural-language question.")
    thread_id: str = Field(
        default="default",
        description="Checkpoint thread id; reuse to resume a conversation.",
    )
    session_id: str | None = Field(
        default=None,
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
