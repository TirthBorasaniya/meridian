"""API route handlers for query, health, and evaluation summary."""

from fastapi import APIRouter

from meridian.api.schemas import (
    EvalSummaryResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
)
from meridian.config import get_settings
from meridian.evaluation.ragas_runner import load_latest_summary
from meridian.graph.graph import run_query
from meridian.ingestion.indexer import count_points

router = APIRouter()


def _to_documents(graded_doc_list: list[dict]) -> list[RetrievedDocument]:
    """Map graded document dicts to response models."""
    document_list: list[RetrievedDocument] = []
    for doc in graded_doc_list:
        payload_dict = doc.get("payload", {})
        document_list.append(
            RetrievedDocument(
                chunk_id=doc.get("chunk_id", ""),
                arxiv_id=payload_dict.get("arxiv_id", ""),
                title=payload_dict.get("title", ""),
                score=float(doc.get("rerank_score", 0.0)),
            )
        )
    return document_list


@router.post("/query", response_model=QueryResponse)
def post_query(request: QueryRequest) -> QueryResponse:
    """Run the agentic RAG graph for a single query."""
    final_state = run_query(
        request.query, thread_id=request.thread_id, session_id=request.session_id
    )
    graded_doc_list = final_state.get("graded_docs", [])
    return QueryResponse(
        query=request.query,
        answer=final_state.get("generation", ""),
        source=final_state.get("source", "corpus"),
        query_type=final_state.get("query_type", ""),
        num_documents=len(graded_doc_list),
        iteration_count=int(final_state.get("iteration_count", 0)),
        documents=_to_documents(graded_doc_list),
    )


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report service readiness and the number of indexed points."""
    settings = get_settings()
    try:
        indexed_points = count_points()
        status = "ok"
    except (RuntimeError, ValueError, OSError):
        indexed_points = 0
        status = "degraded"
    return HealthResponse(
        status=status,
        collection=settings.qdrant_collection,
        indexed_points=indexed_points,
    )


@router.get("/eval-summary", response_model=EvalSummaryResponse)
def get_eval_summary() -> EvalSummaryResponse:
    """Return the most recent persisted RAGAS evaluation summary."""
    summary_dict = load_latest_summary()
    if summary_dict is None:
        return EvalSummaryResponse(available=False)
    return EvalSummaryResponse(
        available=True,
        timestamp=summary_dict.get("timestamp"),
        scores=summary_dict.get("scores", {}),
    )
