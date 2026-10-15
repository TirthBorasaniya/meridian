"""Environment-backed settings: secrets, service endpoints, and filesystem roots.

Everything that is not a secret or a deployment endpoint lives in the YAML files under configs/.
"""

import os
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed process settings loaded from the environment and an optional `.env` file."""

    model_config = SettingsConfigDict(
        # MERIDIAN_ENV_FILE lets tests and CI point away from a developer's .env
        env_file=os.environ.get("MERIDIAN_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============= Secrets =============
    groq_api_key: SecretStr = Field(default=SecretStr(""), alias="GROQ_API_KEY")
    tavily_api_key: SecretStr = Field(default=SecretStr(""), alias="TAVILY_API_KEY")
    hf_token: SecretStr = Field(default=SecretStr(""), alias="HF_TOKEN")
    langfuse_public_key: SecretStr = Field(default=SecretStr(""), alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""), alias="LANGFUSE_SECRET_KEY")
    qdrant_api_key: SecretStr = Field(default=SecretStr(""), alias="QDRANT_API_KEY")

    # ============= Endpoints =============
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    # an empty QDRANT_URL selects embedded local mode at QDRANT_PATH
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_path: str = Field(default="data/qdrant_local", alias="QDRANT_PATH")
    # an empty REDIS_URL selects in-process caches
    redis_url: str = Field(default="", alias="REDIS_URL")

    # ============= Hub repositories =============
    hf_index_repo: str = Field(default="", alias="HF_INDEX_REPO")
    hf_model_repo: str = Field(default="", alias="HF_MODEL_REPO")
    hf_space_repo: str = Field(default="", alias="HF_SPACE_REPO")

    # ============= Runtime =============
    serving_profile: str = Field(default="serving", alias="MERIDIAN_SERVING_PROFILE")
    llm_cache_path: str = Field(default=".cache/llm_cache.sqlite", alias="LLM_CACHE_PATH")
    service_name: str = Field(default="meridian", alias="MERIDIAN_SERVICE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    demo_url: str = Field(default="", alias="MERIDIAN_DEMO_URL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
