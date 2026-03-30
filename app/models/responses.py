"""
Pydantic response models for all API endpoints.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── /health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when all systems are healthy")
    vector_store_count: int = Field(description="Number of indexed candidates in ChromaDB")
    embedding_model: str = Field(description="Embedding model currently loaded")
    llm_model: str = Field(description="LLM model used for query rewriting")
    environment: str


# ─── /ingest ──────────────────────────────────────────────────────────────────

class IngestStartedResponse(BaseModel):
    message: str = Field(description="Human-readable status message")
    status: str = Field(description="'started' or 'already_running'")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Ingestion pipeline started in background.", "status": "started"}
            ]
        }
    }


class IngestStatusResponse(BaseModel):
    status: str = Field(description="idle | running | completed | failed")
    total: int = Field(description="Total candidates in the source database")
    processed: int = Field(description="Candidates successfully embedded and indexed")
    failed: int = Field(description="Candidates that failed to process")
    progress_pct: float = Field(description="Completion percentage (0–100)")
    elapsed_seconds: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None, description="Error message if status == failed")
    force_rebuild: bool = Field(description="Whether this run was a full rebuild")


# ─── /chat ────────────────────────────────────────────────────────────────────

class ExpertResult(BaseModel):
    """
    A single expert match returned by the search engine.
    """
    candidate_id: str = Field(description="UUID of the expert in the source database")
    full_name: str
    email: str = Field(description="Contact email address")
    headline: str = Field(description="Professional headline / tagline")
    current_title: str = Field(description="Most recent job title")
    current_company: str = Field(description="Most recent employer")
    city: str
    country: str
    nationality: str
    years_of_experience: int
    industries: list[str] = Field(description="Industries from all work experience")
    skill_names: list[str] = Field(description="All skills listed on the profile")
    education_summary: str = Field(description="Degrees, fields, and graduation years")
    languages: list[str] = Field(description="Languages and proficiency levels")
    match_score: float = Field(
        description="Cosine similarity score ∈ [0, 1] — higher is more relevant",
        ge=0.0,
        le=1.0,
    )
    rank: int = Field(description="1-based position in the result list")
    explanation: str = Field(
        description=(
            "LLM-generated one-sentence explanation of why this expert matches "
            "your query (empty string if explain=false was requested)"
        )
    )


class ChatResponse(BaseModel):
    """
    Response from POST /chat.
    """
    conversation_id: str = Field(
        description=(
            "Use this ID in subsequent requests to continue the conversation "
            "and enable follow-up queries."
        )
    )
    summary: str = Field(
        description="Conversational summary sentence from the AI assistant"
    )
    intent: str = Field(
        description="The search intent as understood by the AI assistant"
    )
    search_text: str = Field(
        description=(
            "The expanded query text actually used for semantic search "
            "(useful for debugging relevance)"
        )
    )
    total_found: int = Field(
        description=(
            "Total number of matching experts found (may be more than the "
            "number returned if top_k is set)"
        )
    )
    results: list[ExpertResult] = Field(description="Ordered list of matching experts")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "summary": "I found 8 regulatory affairs specialists with pharmaceutical industry experience in the Middle East.",
                    "intent": "Find regulatory affairs experts with pharmaceutical industry experience in the Middle East region",
                    "search_text": "Senior regulatory affairs specialist pharmaceutical drug approval GCC Middle East Saudi Arabia UAE ICH guidelines",
                    "total_found": 8,
                    "results": [
                        {
                            "candidate_id": "70222c8e-2b7a-4a9e-bc42-9ae3eaa2a89a",
                            "full_name": "Sara Ali",
                            "email": "saraali@example.com",
                            "headline": "Regulatory Affairs Director | Pharmaceutical | GCC",
                            "current_title": "Regulatory Affairs Director",
                            "current_company": "Novartis Saudi Arabia",
                            "city": "Riyadh",
                            "country": "Saudi Arabia",
                            "nationality": "Lebanese",
                            "years_of_experience": 16,
                            "industries": ["Pharmaceutical", "Healthcare"],
                            "skill_names": ["Regulatory Affairs", "Drug Registration", "ICH Guidelines"],
                            "education_summary": "MSc in Pharmaceutical Sciences from AUB (2009)",
                            "languages": ["English (Native)", "Arabic (Native)", "French (Intermediate)"],
                            "match_score": 0.923,
                            "rank": 1,
                            "explanation": "Sara has 16 years in pharmaceutical regulatory affairs with direct GCC drug registration experience.",
                        }
                    ],
                }
            ]
        }
    }


# ─── /experts/{id} ───────────────────────────────────────────────────────────

class ExpertDetailResponse(BaseModel):
    """Full expert profile including raw profile text."""
    candidate_id: str
    full_name: str
    email: str
    headline: str
    current_title: str
    current_company: str
    city: str
    country: str
    nationality: str
    years_of_experience: int
    industries: list[str]
    skill_names: list[str]
    education_summary: str
    languages: list[str]
    profile_text: str = Field(description="The full text document used for semantic embedding")


# ─── /conversations ───────────────────────────────────────────────────────────

class ConversationInfo(BaseModel):
    conversation_id: str
    message_count: int
    last_active: float = Field(description="Unix timestamp of last activity")


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    message_count: int
    created_at: float
    last_active: float
    has_results: bool
    messages: list[dict[str, Any]] = Field(
        description="Full message history [{role, content}]"
    )


class DeleteResponse(BaseModel):
    message: str
