"""FastAPI service: `POST /v1/ask` (SSE), `POST /v1/search`, `GET /healthz`, `GET /readyz`,
`GET /metrics`, `GET /v1/index`, and the single-page UI at `/`.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from importlib import resources
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from meridian.common.metrics import REGISTRY
from meridian.serving.runtime import Runtime, build_runtime, index_info_dict
from meridian.serving.sse import sse_event, stream_blocking

logger = logging.getLogger(__name__)

MAX_IDENTIFIER_LENGTH = 128
MAX_FILTER_PAPERS = 20


# ============= Schemas =============


class AskRequest(BaseModel):
    """Body of `/v1/ask`."""

    question: str = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    paper_filter: list[str] | None = Field(default=None, max_length=MAX_FILTER_PAPERS)


class SearchRequest(BaseModel):
    """Body of `/v1/search`."""

    query: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=30)
    mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    paper_filter: list[str] | None = Field(default=None, max_length=MAX_FILTER_PAPERS)


# ============= Helpers =============


def client_key(request: Request) -> str:
    """Return the client address; behind a proxy, uvicorn's proxy headers set it."""
    return request.client.host if request.client else "unknown"


def enforce_limits(runtime: Runtime, request: Request, text: str) -> None:
    """Reject oversized input and clients over their rate limit."""
    if len(text) > runtime.serving_config.max_question_chars:
        raise HTTPException(status_code=413, detail="question too long")
    decision = runtime.rate_limiter.check(client_key(request))
    if not decision.o_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded ({decision.window})",
            headers={"Retry-After": str(decision.retry_after_s)},
        )


# ============= App =============


def create_app(runtime: Runtime | None = None) -> FastAPI:
    """
    Create the application.

    Parameters
    ----------
    runtime : Runtime or None
        Injected runtime for tests; built from settings at startup when None.

    Returns
    -------
    app : FastAPI
        The application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or build_runtime()
        yield

    app = FastAPI(title="Meridian", version="0.2.0", lifespan=lifespan)
    ui_root = resources.files("meridian.serving").joinpath("ui")
    if ui_root.joinpath("index.html").is_file():
        app.mount("/static", StaticFiles(directory=str(ui_root)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> HTMLResponse:
        page = ui_root.joinpath("index.html")
        if not page.is_file():
            return HTMLResponse("<p>Meridian API. See /docs.</p>")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request) -> JSONResponse:
        info = request.app.state.runtime.index_info(o_refresh=True)
        ready = info.collection is not None and info.points > 0
        return JSONResponse(
            {"ready": ready, "collection": info.collection, "points": info.points},
            status_code=200 if ready else 503,
        )

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/index")
    def index_endpoint(request: Request) -> dict:
        return index_info_dict(request.app.state.runtime.index_info())

    @app.post("/v1/search")
    def search(body: SearchRequest, request: Request) -> dict:
        runtime = request.app.state.runtime
        enforce_limits(runtime, request, body.query)
        return runtime.search(body.query, body.k, body.mode, body.paper_filter)

    @app.post("/v1/ask")
    async def ask(body: AskRequest, request: Request) -> StreamingResponse:
        runtime = request.app.state.runtime
        enforce_limits(runtime, request, body.question)
        session_id = body.session_id or uuid.uuid4().hex

        async def event_stream():
            yield sse_event("meta", {"session_id": session_id})
            async for kind, value in stream_blocking(
                lambda emit: runtime.ask(body.question, session_id, body.paper_filter, emit)
            ):
                if kind == "event":
                    yield sse_event(value.get("type", "token"), value)
                elif kind == "result":
                    yield sse_event("final", value)
                else:
                    logger.error("ask failed", extra={"error": str(value)})
                    yield sse_event("error", {"detail": "the answer could not be generated"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def cli_serve(args) -> int:
    """`meridian serve`: run uvicorn."""
    import uvicorn

    from meridian.common.settings import get_settings

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips=get_settings().forwarded_allow_ips,
        log_config=None,
    )
    return 0
