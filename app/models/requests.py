"""
Pydantic request models for all API endpoints.
"""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator


# ─── /ingest ──────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Request body for POST /ingest.

    Triggering ingestion without a body (empty JSON ``{}``) uses all defaults.
    """
    force_rebuild: bool = Field(
        default=False,
        description=(
            "If true, drop the existing ChromaDB collection and re-index all "
            "candidates from scratch. Useful when the embedding model changes. "
            "If false (default), upsert — safe to run multiple times."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"force_rebuild": False},
                {"force_rebuild": True},
            ]
        }
    }


# ─── /chat ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    Minimal required field is ``query``. All other fields are optional and
    have sensible defaults.
    """
    query: Annotated[str, Field(
        min_length=3,
        max_length=2000,
        description="Natural language search query or follow-up question.",
        examples=[
            "Find regulatory affairs experts with pharma experience in the Middle East",
            "Show me senior data scientists with Python and ML skills in Germany",
            "Filter those to only people with more than 10 years of experience",
        ],
    )]

    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "UUID of an existing conversation. Provide this to continue a "
            "previous search session so follow-up queries can reference prior "
            "results. If omitted or null, a new conversation is started and "
            "its ID is returned in the response."
        ),
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Number of experts to return (1–50). Defaults to the server-side "
            "DEFAULT_TOP_K setting (10)."
        ),
    )

    explain: bool = Field(
        default=True,
        description=(
            "If true (default), each result includes an LLM-generated one-sentence "
            "explanation of why this expert matches your query. Set to false to "
            "skip LLM explanation calls and get a faster, cheaper response."
        ),
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find regulatory affairs experts with pharma experience in the Middle East",
                    "top_k": 10,
                    "explain": True,
                },
                {
                    "query": "Filter those to only people based in Saudi Arabia",
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "top_k": 5,
                    "explain": False,
                },
            ]
        }
    }
