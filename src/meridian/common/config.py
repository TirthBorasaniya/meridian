"""Typed loaders for the YAML files under configs/, one pydantic model per file.

Every loaded config remembers its source path and SHA-256 so that results artifacts can record
exactly which configuration produced them.
"""

import hashlib
import os
from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

# ============= Constants =============

CONFIG_DIR_ENV = "MERIDIAN_CONFIG_DIR"
DEFAULT_CONFIG_DIR = "configs"

T = TypeVar("T", bound="VersionedConfig")


# ============= Base =============


class VersionedConfig(BaseModel):
    """Base for every config file: a required `version` plus provenance."""

    model_config = ConfigDict(extra="forbid")

    version: int
    _source_path: str = PrivateAttr(default="")
    _sha256: str = PrivateAttr(default="")

    @property
    def source_path(self) -> str:
        """Path of the YAML file this config was loaded from."""
        return self._source_path

    @property
    def sha256(self) -> str:
        """SHA-256 of the YAML file bytes this config was loaded from."""
        return self._sha256


def config_dir() -> Path:
    """Return the directory holding the YAML configs."""
    return Path(os.environ.get(CONFIG_DIR_ENV, DEFAULT_CONFIG_DIR))


def file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(model: type[T], path: str | Path) -> T:
    """
    Load and validate one YAML config file.

    Parameters
    ----------
    model : type
        The `VersionedConfig` subclass describing the file.
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    config : VersionedConfig
        Validated config with `source_path` and `sha256` set.
    """
    with open(path, encoding="utf-8") as handle:
        raw_dict = yaml.safe_load(handle) or {}
    config = model.model_validate(raw_dict)
    config._source_path = str(path)
    config._sha256 = file_sha256(path)
    return config


# ============= Corpus =============


class BackfillConfig(BaseModel):
    """Initial backfill query parameters."""

    model_config = ConfigDict(extra="forbid")

    date_ceiling: date
    sort_by: Literal["relevance", "submittedDate", "lastUpdatedDate"]
    sort_order: Literal["ascending", "descending"]
    max_results: int = Field(gt=0)
    seed_ids_path: str


class IncrementConfig(BaseModel):
    """Daily incremental query parameters."""

    model_config = ConfigDict(extra="forbid")

    sort_by: Literal["relevance", "submittedDate", "lastUpdatedDate"]
    sort_order: Literal["ascending", "descending"]
    max_papers_per_run: int = Field(gt=0)
    # re-read this many days before the watermark, because arXiv announces with a delay
    lookback_days: int = Field(ge=0)
    # optional clause ANDed into the increment query to narrow a broad topic
    extra_clause: str | None = None
    # re-check corpus papers for newer versions on every increment
    o_check_versions: bool = True


class ArxivQueryConfig(BaseModel):
    """arXiv query definition for the corpus topic."""

    model_config = ConfigDict(extra="forbid")

    api_url: str
    categories: list[str] = Field(min_length=1)
    abstract_terms: list[str] = Field(min_length=1)
    recorded_backfill_query: str
    date_floor: date
    backfill: BackfillConfig
    increment: IncrementConfig
    page_size: int = Field(gt=0, le=2000)


class ThrottleConfig(BaseModel):
    """arXiv request pacing and retry policy."""

    model_config = ConfigDict(extra="forbid")

    min_interval_seconds: float = Field(ge=3.0)
    jitter_seconds: float = Field(ge=0.0)
    max_retries: int = Field(ge=0)
    backoff_base_seconds: float = Field(gt=0.0)
    backoff_max_seconds: float = Field(gt=0.0)
    request_timeout_seconds: float = Field(gt=0.0)
    user_agent: str


class CorpusPaths(BaseModel):
    """Filesystem locations for corpus artifacts."""

    model_config = ConfigDict(extra="forbid")

    pdf_dir: str
    parsed_dir: str
    manifest: str


class CorpusConfig(VersionedConfig):
    """configs/corpus.yaml."""

    topic_name: str
    topic_description: str
    arxiv: ArxivQueryConfig
    throttle: ThrottleConfig
    paths: CorpusPaths


# ============= Index =============


class ParserConfig(BaseModel):
    """PDF parsing options that determine the parsed text."""

    model_config = ConfigDict(extra="forbid")

    parser_version: int = Field(ge=1)
    references_headings: list[str]
    repeated_line_page_fraction: float = Field(gt=0.0, le=1.0)
    repeated_line_edge_lines: int = Field(ge=1)


class ChunkerConfig(BaseModel):
    """Section-aware token chunker options."""

    model_config = ConfigDict(extra="forbid")

    chunk_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    min_section_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> "ChunkerConfig":
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        return self


class DenseConfig(BaseModel):
    """Dense embedding model options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = Field(gt=0)


class SparseConfig(BaseModel):
    """Sparse embedding model options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str


class RerankerConfig(BaseModel):
    """Cross-encoder options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    batch_size: int = Field(gt=0)


class RetrievalConfig(BaseModel):
    """Retrieval depths at each stage."""

    model_config = ConfigDict(extra="forbid")

    prefetch_limit: int = Field(gt=0)
    fusion_limit: int = Field(gt=0)
    rerank_depth: int = Field(gt=0)
    final_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_depths(self) -> "RetrievalConfig":
        if self.final_k > self.rerank_depth:
            raise ValueError("final_k must not exceed rerank_depth")
        if self.rerank_depth > self.fusion_limit:
            raise ValueError("rerank_depth must not exceed fusion_limit")
        return self


class IndexConfig(VersionedConfig):
    """configs/index.yaml."""

    collection_prefix: str
    alias: str
    keep_previous_versions: int = Field(ge=0)
    upsert_batch_size: int = Field(gt=0)
    parser: ParserConfig
    chunker: ChunkerConfig
    dense: DenseConfig
    sparse: SparseConfig
    reranker: RerankerConfig
    retrieval: RetrievalConfig

    def build_fingerprint(self) -> dict:
        """Return the settings whose change requires a full rebuild."""
        return {
            "parser": self.parser.model_dump(),
            "chunker": self.chunker.model_dump(),
            "dense_model": self.dense.model,
            "dense_document_prefix": self.dense.document_prefix,
            "sparse_model": self.sparse.model,
        }


# ============= Models =============


class RoleConfig(BaseModel):
    """One LLM role: model and sampling parameters."""

    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class PriceConfig(BaseModel):
    """USD per million tokens for one model."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)


class ModelsConfig(VersionedConfig):
    """configs/models.yaml."""

    provider: Literal["groq"]
    request_timeout_seconds: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)
    backoff_base_seconds: float = Field(gt=0.0)
    backoff_max_seconds: float = Field(gt=0.0)
    roles: dict[str, RoleConfig]
    prices_usd_per_million_tokens: dict[str, PriceConfig]
    prices_source: str
    prices_checked_on: date | None = None

    @model_validator(mode="after")
    def _check_roles(self) -> "ModelsConfig":
        required_set = {
            "router",
            "rewriter",
            "grader",
            "generator",
            "judge",
            "question_writer",
            "pair_writer",
        }
        missing_set = required_set - set(self.roles)
        if missing_set:
            raise ValueError(f"models.yaml is missing roles: {sorted(missing_set)}")
        if self.roles["judge"].model == self.roles["generator"].model:
            raise ValueError("the judge model must differ from the generator model")
        if "llama" in self.roles["pair_writer"].model.lower():
            raise ValueError("training pairs must not be generated by a Llama model")
        return self

    def role(self, name: str) -> RoleConfig:
        """Return the config of one role."""
        return self.roles[name]


# ============= Gate =============


class GateConfig(VersionedConfig):
    """configs/gate.yaml."""

    max_regression: float = Field(ge=0.0)
    faithfulness_max_regression: float = Field(ge=0.0)
    fast_metric: str
    full_metrics: list[str] = Field(min_length=1)
    baseline_path: str
    retrieval_set_path: str
    answer_set_path: str


# ============= Agent =============


class GradingConfig(BaseModel):
    """Score-band grading."""

    model_config = ConfigDict(extra="forbid")

    o_score_band: bool
    accept_score: float = Field(ge=0.0, le=1.0)
    reject_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_band(self) -> "GradingConfig":
        if self.reject_score > self.accept_score:
            raise ValueError("reject_score must not exceed accept_score")
        return self


class DecideConfig(BaseModel):
    """Retry and fallback policy."""

    model_config = ConfigDict(extra="forbid")

    min_supporting_chunks: int = Field(ge=1)
    max_retries: int = Field(ge=0)
    o_web_fallback: bool


class VerifyConfig(BaseModel):
    """Citation verification policy."""

    model_config = ConfigDict(extra="forbid")

    sample_rate: float = Field(ge=0.0, le=1.0)
    sample_seed: int
    max_regenerations: int = Field(ge=0)
    regenerate_above_unsupported_fraction: float = Field(ge=0.0, le=1.0)


class WebConfig(BaseModel):
    """Web search fallback."""

    model_config = ConfigDict(extra="forbid")

    max_results: int = Field(gt=0)


class ConversationConfig(BaseModel):
    """Conversation history passed to the router and rewriter."""

    model_config = ConfigDict(extra="forbid")

    history_messages: int = Field(ge=0)


class AgentConfig(VersionedConfig):
    """configs/agent.yaml."""

    retrieval_mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
    grading: GradingConfig
    decide: DecideConfig
    verify: VerifyConfig
    web: WebConfig
    conversation: ConversationConfig
    excerpt_sentences: int = Field(gt=0)
    checkpoint_path: str


# ============= Scheduler =============


class SchedulerConfig(VersionedConfig):
    """configs/scheduler.yaml."""

    timezone: str
    daily_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    misfire_grace_seconds: int = Field(gt=0)
    deadline_minutes: float = Field(gt=0)


# ============= Cached accessors =============


@cache
def get_corpus_config() -> CorpusConfig:
    """Return configs/corpus.yaml."""
    return load_config(CorpusConfig, config_dir() / "corpus.yaml")


@cache
def get_index_config() -> IndexConfig:
    """Return configs/index.yaml."""
    return load_config(IndexConfig, config_dir() / "index.yaml")


@cache
def get_models_config() -> ModelsConfig:
    """Return configs/models.yaml."""
    return load_config(ModelsConfig, config_dir() / "models.yaml")


@cache
def get_gate_config() -> GateConfig:
    """Return configs/gate.yaml."""
    return load_config(GateConfig, config_dir() / "gate.yaml")


@cache
def get_agent_config() -> AgentConfig:
    """Return configs/agent.yaml."""
    return load_config(AgentConfig, config_dir() / "agent.yaml")


@cache
def get_scheduler_config() -> SchedulerConfig:
    """Return configs/scheduler.yaml."""
    return load_config(SchedulerConfig, config_dir() / "scheduler.yaml")


def clear_config_caches() -> None:
    """Clear every cached config accessor, so tests can swap config files."""
    for accessor in (
        get_corpus_config,
        get_index_config,
        get_models_config,
        get_gate_config,
        get_scheduler_config,
        get_agent_config,
    ):
        accessor.cache_clear()
