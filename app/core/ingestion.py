"""
Data ingestion pipeline.

Orchestrates the full ETL flow:
  PostgreSQL → profile text documents → embeddings → ChromaDB

Design decisions
────────────────
1. **Single embedding per candidate** (not chunked by section):
   A candidate's full expertise is best captured as one holistic vector.
   Splitting into "skills chunk" + "experience chunk" would require
   multi-vector retrieval and re-ranking — unnecessary complexity for
   profiles of this size (~200–600 tokens each).

2. **Batch processing** to keep memory usage flat:
   We process ``INGEST_BATCH_SIZE`` (default 100) candidates at a time,
   fetching sub-tables only for that batch, so RAM consumption is O(batch)
   not O(all candidates).

3. **Upsert semantics** mean re-running the pipeline is safe (idempotent).
   Changed profiles get their embeddings refreshed; unchanged ones are
   overwritten with the same data (no-op in practice).

4. **Progress tracking via a shared state dict** so the /ingest endpoint
   can stream live status back to callers without polling.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from app.core.embedder import get_embedder
from app.core.postgres import (
    build_full_profiles,
    fetch_all_candidates,
    fetch_education,
    fetch_languages,
    fetch_skills,
    fetch_total_candidate_count,
    fetch_work_experience,
)
from app.core.vector_store import VectorStore
from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Progress state ───────────────────────────────────────────────────────────

@dataclass
class IngestionStatus:
    """
    Mutable state object updated throughout the pipeline run.
    Exposed via GET /ingest/status so callers can poll progress.
    """
    status: str = "idle"           # idle | running | completed | failed
    total: int = 0
    processed: int = 0
    failed: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    force_rebuild: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return round(end - self.started_at, 1)

    @property
    def progress_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return round(100 * self.processed / self.total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "failed": self.failed,
            "progress_pct": self.progress_pct,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "force_rebuild": self.force_rebuild,
        }


# Module-level singleton so the FastAPI route can read it
_current_status = IngestionStatus()


def get_ingestion_status() -> IngestionStatus:
    return _current_status


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_ingestion(force_rebuild: bool = False) -> IngestionStatus:
    """
    Execute the full ingestion pipeline synchronously.

    Args:
        force_rebuild: If True, drop and recreate the ChromaDB collection
                       before ingesting (full re-index).

    Returns:
        The final IngestionStatus after the pipeline completes.
    """
    global _current_status

    settings = get_settings()
    batch_size = settings.ingest_batch_size

    _current_status = IngestionStatus(
        status="running",
        started_at=time.time(),
        force_rebuild=force_rebuild,
    )

    try:
        vector_store = VectorStore()
        embedder = get_embedder()

        if force_rebuild:
            logger.info("Force rebuild requested — clearing existing collection.")
            vector_store.delete_all()

        # ── Step 1: fetch base candidates ─────────────────────────────────────
        logger.info("Fetching candidate base records from PostgreSQL…")
        all_candidates = fetch_all_candidates()
        _current_status.total = len(all_candidates)
        logger.info("Found %d candidates to process.", _current_status.total)

        # ── Step 2: process in batches ────────────────────────────────────────
        for batch_start in tqdm(
            range(0, len(all_candidates), batch_size),
            desc="Ingesting batches",
            unit="batch",
        ):
            batch = all_candidates[batch_start : batch_start + batch_size]
            candidate_ids = [str(c["candidate_id"]) for c in batch]

            try:
                # Fetch all sub-table data for this batch in parallel queries
                work_map = fetch_work_experience(candidate_ids)
                skills_map = fetch_skills(candidate_ids)
                edu_map = fetch_education(candidate_ids)
                lang_map = fetch_languages(candidate_ids)

                # Build rich profile dicts + profile_text
                profiles = build_full_profiles(
                    batch, work_map, skills_map, edu_map, lang_map
                )

                # Generate embeddings for this batch
                texts = [p["profile_text"] for p in profiles]
                embeddings = embedder.embed_batch(
                    texts,
                    batch_size=min(64, batch_size),
                    show_progress=False,
                )

                # Upsert into ChromaDB
                vector_store.upsert_profiles(profiles, embeddings)
                _current_status.processed += len(batch)

            except Exception as e:
                logger.error(
                    "Batch %d–%d failed: %s",
                    batch_start,
                    batch_start + len(batch),
                    e,
                    exc_info=True,
                )
                _current_status.failed += len(batch)

        # ── Step 3: finalise ──────────────────────────────────────────────────
        _current_status.status = "completed"
        _current_status.finished_at = time.time()
        logger.info(
            "Ingestion complete: %d processed, %d failed in %.1f s",
            _current_status.processed,
            _current_status.failed,
            _current_status.elapsed_seconds,
        )

    except Exception as exc:
        _current_status.status = "failed"
        _current_status.error = str(exc)
        _current_status.finished_at = time.time()
        logger.exception("Ingestion pipeline failed: %s", exc)

    return _current_status
