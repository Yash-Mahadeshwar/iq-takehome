"""
Application configuration loaded from environment variables / .env file.
All secrets live in .env — never hard-coded here.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Source database ───────────────────────────────────────────────────────
    postgres_url: str = Field(
        ..., description="PostgreSQL connection string (read-only)"
    )

    # ── OpenRouter / LLM ──────────────────────────────────────────────────────
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_model: str = Field(
        default="anthropic/claude-3-haiku",
        description="Model slug on OpenRouter (e.g. anthropic/claude-3-haiku)",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL (OpenAI-compatible)",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name",
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./chroma_data",
        description="Directory where ChromaDB stores its data on disk",
    )
    chroma_collection_name: str = Field(
        default="expert_profiles",
        description="ChromaDB collection that holds all candidate embeddings",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    ingest_batch_size: int = Field(
        default=100,
        description="Candidates processed per embedding batch",
    )

    # ── Search ────────────────────────────────────────────────────────────────
    default_top_k: int = Field(
        default=10,
        description="Default number of expert results returned per query",
    )
    max_top_k: int = Field(
        default=50,
        description="Hard cap on top_k to prevent over-fetching",
    )

    # ── Conversation ──────────────────────────────────────────────────────────
    conversation_ttl_minutes: int = Field(
        default=60,
        description="Minutes of inactivity before a conversation is evicted",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default="Expert Network Search Copilot")
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
