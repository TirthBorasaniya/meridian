"""Centralised application configuration backed by environment variables.

All secrets, connection strings, model identifiers, and pipeline tunables are
resolved here through pydantic-settings so that no other module reads the
process environment directly.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API credentials ---
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")

    # --- Langfuse observability ---
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    # --- Metadata store (PostgreSQL) ---
    database_url: str = Field(
        default="postgresql+psycopg://meridian:meridian@localhost:5432/meridian",
        validation_alias="DATABASE_URL",
    )

    # --- Qdrant vector store ---
    # When ``qdrant_url`` is empty the client uses local disk persistence at
    # ``qdrant_path``. When set, it targets a Qdrant server.
    # ``qdrant_api_key`` applies only to the server mode. Managed deployments
    # (Qdrant Cloud) reject unauthenticated requests; a local Docker Compose
    # Qdrant has no auth, so the key stays optional and is omitted when empty.
    qdrant_url: str = Field(default="", validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", validation_alias="QDRANT_API_KEY")
    qdrant_path: str = Field(default="data/qdrant_db", validation_alias="QDRANT_PATH")
    qdrant_collection: str = Field(
        default="meridian_papers", validation_alias="QDRANT_COLLECTION"
    )

    # --- Models ---
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", validation_alias="EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base", validation_alias="RERANKER_MODEL"
    )
    grading_model: str = Field(
        default="llama-3.1-8b-instant", validation_alias="GRADING_MODEL"
    )
    generation_model: str = Field(
        default="llama-3.3-70b-versatile", validation_alias="GENERATION_MODEL"
    )

    # --- Checkpointing ---
    checkpoint_db_path: str = Field(
        default="data/checkpoints/graph.db", validation_alias="CHECKPOINT_DB_PATH"
    )

    # --- Cross-session memory (distinct from in-session SqliteSaver checkpointing) ---
    session_db_path: str = Field(
        default="data/checkpoints/sessions.db", validation_alias="SESSION_DB_PATH"
    )
    session_context_turns: int = Field(default=6, validation_alias="SESSION_CONTEXT_TURNS")

    # --- Corpus and ingestion ---
    arxiv_categories: tuple[str, ...] = ("cs.CL", "cs.LG", "cs.AI")
    arxiv_topic_terms: tuple[str, ...] = (
        "large language model",
        "chain-of-thought",
        "reasoning",
        "evaluation",
        "benchmark",
        "calibration",
    )
    arxiv_start_year: int = 2022
    arxiv_end_year: int = 2024
    arxiv_max_requests_per_second: float = 3.0
    corpus_raw_dir: str = "data/corpus/raw"
    corpus_parsed_dir: str = "data/corpus/parsed"
    target_corpus_size: int = 200

    # --- Chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- Retrieval ---
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5
    rerank_score_threshold: float = 0.5

    # --- Graph control ---
    max_iterations: int = 3

    # --- API request limits ---
    # ``/query`` is unauthenticated and calls paid Groq and Tavily APIs, so
    # both the per-request size and the per-client request rate are bounded.
    # Set ``RATE_LIMIT_REQUESTS`` to 0 to disable rate limiting.
    max_query_length: int = Field(default=2000, validation_alias="MAX_QUERY_LENGTH")
    rate_limit_requests: int = Field(default=20, validation_alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(
        default=60, validation_alias="RATE_LIMIT_WINDOW_SECONDS"
    )

    # --- Embedding dimensionality of BAAI/bge-small-en-v1.5 ---
    embedding_dim: int = 384

    # --- BGE asymmetric instruction prefix applied to queries only ---
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Returns
    -------
    Settings
        The singleton settings object resolved from the environment.
    """
    return Settings()
