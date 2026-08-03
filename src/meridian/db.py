"""PostgreSQL access layer for arXiv paper metadata and ingestion state.

Qdrant holds chunk embeddings; this relational store holds canonical paper
metadata and per-paper ingestion status so that pipeline re-runs are
idempotent. The ``Paper`` row for an arxiv_id is upserted as the paper moves
through the fetch, parse, and index stages.

This module is not part of the originally planned module layout; it exists
because the corpus specification mandates a PostgreSQL metadata store and the
session factory is shared across ingestion tasks.
"""

from datetime import datetime
from functools import lru_cache

from sqlalchemy import String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from meridian.config import get_settings

# Ingestion status values, in order of pipeline progression.
STATUS_PENDING = "pending"
STATUS_FETCHED = "fetched"
STATUS_PARSED = "parsed"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
# Terminal and distinct from "failed": arXiv serves no PDF for a withdrawn
# version, so the paper is excluded by design rather than by a fault that a
# re-run could clear.
STATUS_WITHDRAWN = "withdrawn"


class Base(DeclarativeBase):
    """Declarative base for all Meridian ORM models."""


class Paper(Base):
    """Canonical metadata and ingestion state for a single arXiv paper."""

    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[str] = mapped_column(Text)  # semicolon-joined author names
    abstract: Mapped[str] = mapped_column(Text)
    categories: Mapped[str] = mapped_column(Text)  # space-joined arXiv categories
    published: Mapped[datetime] = mapped_column()
    pdf_path: Mapped[str] = mapped_column(Text, default="")
    parsed_path: Mapped[str] = mapped_column(Text, default="")
    ingestion_status: Mapped[str] = mapped_column(String(32), default=STATUS_PENDING)
    chunk_count: Mapped[int] = mapped_column(default=0)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine bound to ``DATABASE_URL``.

    Returns
    -------
    sqlalchemy.engine.Engine
        Engine with pre-ping enabled to recover stale pooled connections.
    """
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    """Return a cached session factory bound to the engine.

    Returns
    -------
    sqlalchemy.orm.sessionmaker
        Factory producing sessions with autoflush disabled.
    """
    return sessionmaker(
        bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
    )


def init_db() -> None:
    """Create all tables defined on :class:`Base` if they do not yet exist."""
    Base.metadata.create_all(get_engine())
