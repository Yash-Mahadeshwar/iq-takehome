"""
POST /ingest      — trigger the ingestion pipeline (runs in background thread)
GET  /ingest/status — poll ingestion progress
"""
from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException

from app.core.ingestion import get_ingestion_status, run_ingestion
from app.models.requests import IngestRequest
from app.models.responses import IngestStartedResponse, IngestStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestStartedResponse,
    status_code=202,
    summary="Trigger ingestion pipeline",
    description=(
        "Starts the ETL pipeline: extracts candidate profiles from PostgreSQL, "
        "generates sentence-transformer embeddings, and upserts them into ChromaDB. "
        "\n\n"
        "**This is an async fire-and-forget operation** — the endpoint returns "
        "immediately with HTTP 202 Accepted. Poll `GET /ingest/status` to monitor "
        "progress.\n\n"
        "Running the pipeline while one is already in progress returns HTTP 409.\n\n"
        "Set `force_rebuild=true` to drop and fully re-index the collection "
        "(needed if you change the embedding model)."
    ),
)
def trigger_ingestion(body: IngestRequest = IngestRequest()) -> IngestStartedResponse:
    status = get_ingestion_status()

    if status.status == "running":
        raise HTTPException(
            status_code=409,
            detail="An ingestion pipeline is already running. Poll GET /ingest/status for progress.",
        )

    def _run() -> None:
        run_ingestion(force_rebuild=body.force_rebuild)

    t = threading.Thread(target=_run, daemon=True, name="ingestion-worker")
    t.start()

    return IngestStartedResponse(
        message="Ingestion pipeline started in background. Poll GET /ingest/status for progress.",
        status="started",
    )


@router.get(
    "/ingest/status",
    response_model=IngestStatusResponse,
    summary="Poll ingestion progress",
    description=(
        "Returns the current state of the ingestion pipeline:\n\n"
        "- `idle` — no pipeline has been run yet.\n"
        "- `running` — pipeline is currently processing candidates.\n"
        "- `completed` — pipeline finished successfully.\n"
        "- `failed` — pipeline encountered a fatal error (see `error` field).\n\n"
        "Poll every few seconds while `status == 'running'`."
    ),
)
def ingestion_status() -> IngestStatusResponse:
    s = get_ingestion_status()
    return IngestStatusResponse(**s.to_dict())
