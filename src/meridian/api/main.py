"""FastAPI application entrypoint.

The lifespan context manager warms the heavy singletons (embedder,
cross-encoder, compiled graph) and ensures the Qdrant collection exists before
the server accepts traffic, so the first request does not pay the model-load
cost.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from meridian.api.routes import router
from meridian.graph.graph import get_graph
from meridian.ingestion.embedder import get_embedder
from meridian.ingestion.indexer import ensure_collection
from meridian.retrieval.reranker import get_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup and release them on shutdown."""
    # Startup: ensure the vector collection exists and warm the model singletons.
    ensure_collection()
    get_embedder()
    get_reranker()
    get_graph()
    yield
    # Shutdown: clients are process-lived and require no explicit teardown.


app = FastAPI(
    title="Meridian",
    description="Agentic RAG over an arXiv corpus of LLM reasoning and evaluation papers.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)
