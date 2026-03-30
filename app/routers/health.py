"""
GET /health — liveness and readiness check.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import get_settings
from app.core.embedder import get_embedder
from app.core.vector_store import VectorStore
from app.models.responses import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the current status of all system components: "
        "ChromaDB collection size, embedding model, and LLM configuration."
    ),
)
def health_check() -> HealthResponse:
    settings = get_settings()

    try:
        vs = VectorStore()
        count = vs.count()
    except Exception as e:
        logger.warning("ChromaDB unavailable during health check: %s", e)
        count = -1

    try:
        embedder = get_embedder()
        emb_model = settings.embedding_model
    except Exception as e:
        logger.warning("Embedder unavailable during health check: %s", e)
        emb_model = f"ERROR: {e}"

    return HealthResponse(
        status="ok",
        vector_store_count=count,
        embedding_model=emb_model,
        llm_model=settings.openrouter_model,
        environment=settings.environment,
    )
