"""
ChromaDB vector store wrapper.

Responsibilities
────────────────
- Initialise / open the persistent ChromaDB collection.
- Upsert candidate embeddings + metadata.
- Run similarity searches and return structured hits.

Why ChromaDB?
─────────────
- Zero infrastructure: runs in-process, stores on local disk.
- Persistent across restarts (no re-ingestion needed after first run).
- Native cosine-similarity support with ``hnsw:space=cosine``.
- Handles 10 K–100 K profiles without performance issues.
- Rich metadata filtering (``where`` clause) for post-search faceting.
- Easy Python API; no Docker or managed service required.

Alternatives considered:
- pgvector   — great if already on Postgres, requires schema changes.
- Pinecone   — managed, but adds external dependency and cost.
- Qdrant     — excellent, but requires running a separate Docker container.
- FAISS      — fast, but in-memory only (no persistence) and no metadata.
"""
from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)

# Maximum metadata string length (ChromaDB has a ~1 MB per-doc limit but
# we keep individual fields short to stay well within that).
_MAX_META_STR = 2000


def _truncate(value: Any, max_len: int = _MAX_META_STR) -> str:
    """Convert a value to string and cap its length."""
    s = str(value) if value is not None else ""
    return s[:max_len]


class VectorStore:
    """
    Wrapper around a single ChromaDB collection that stores expert embeddings.

    Metadata schema stored per document
    ────────────────────────────────────
    candidate_id       — UUID string (primary key)
    full_name          — "First Last"
    email              — contact email
    headline           — LinkedIn-style headline
    city               — current city
    country            — current country
    nationality        — nationality
    years_of_experience— integer (stored as int)
    current_title      — most recent job title
    current_company    — most recent company name
    industries         — comma-separated list of industries
    skill_names        — comma-separated list of skill names
    skill_categories   — comma-separated list of skill categories
    education_summary  — comma-separated "Degree in Field (Year)"
    languages          — comma-separated "Language (Proficiency)"
    profile_text       — full text used for embedding (for display / debugging)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' opened (%d docs)",
            settings.chroma_collection_name,
            self._collection.count(),
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_profiles(
        self,
        profiles: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert a batch of candidate profiles with their precomputed embeddings.

        Using ``upsert`` (not ``add``) means re-ingestion is idempotent —
        running POST /ingest twice won't duplicate records.
        """
        if not profiles:
            return

        ids: list[str] = []
        metas: list[dict] = []
        documents: list[str] = []

        for profile in profiles:
            cid = str(profile["candidate_id"])
            ids.append(cid)
            documents.append(profile.get("profile_text", ""))

            # Build flat metadata dict (ChromaDB only accepts str/int/float/bool)
            edu_summary = ", ".join(
                f"{e['degree']} in {e['field_of_study']}"
                + (f" ({e['graduation_year']})" if e.get("graduation_year") else "")
                for e in profile.get("education", [])
            )
            lang_summary = ", ".join(
                f"{l['language']} ({l['proficiency']})" if l.get("proficiency") else l["language"]
                for l in profile.get("languages", [])
            )

            meta = {
                "candidate_id": cid,
                "full_name": _truncate(profile.get("full_name", "")),
                "email": _truncate(profile.get("email", "")),
                "headline": _truncate(profile.get("headline", ""), 500),
                "city": _truncate(profile.get("city", "")),
                "country": _truncate(profile.get("country", "")),
                "nationality": _truncate(profile.get("nationality", "")),
                "years_of_experience": int(profile.get("years_of_experience") or 0),
                "current_title": _truncate(profile.get("current_title", "")),
                "current_company": _truncate(profile.get("current_company", "")),
                "industries": _truncate(", ".join(profile.get("industries", []))),
                "skill_names": _truncate(", ".join(profile.get("skill_names", []))),
                "skill_categories": _truncate(", ".join(profile.get("skill_categories", []))),
                "education_summary": _truncate(edu_summary),
                "languages": _truncate(lang_summary),
                "profile_text": _truncate(profile.get("profile_text", ""), _MAX_META_STR),
            }
            metas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metas,
            documents=documents,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run cosine-similarity search and return a list of result dicts.

        Each result contains:
            - All metadata fields (candidate info)
            - ``score``     : cosine similarity ∈ [0, 1] (higher = better)
            - ``document``  : the profile text used for embedding
            - ``rank``      : 1-based result position
        """
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(1, self._collection.count())),
            "include": ["metadatas", "distances", "documents"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        hits = []
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]

        for rank, (meta, dist, doc) in enumerate(zip(metadatas, distances, documents), start=1):
            # ChromaDB cosine distance = 1 - cosine_similarity
            score = round(1.0 - dist, 4)
            hit = {**meta, "score": score, "document": doc, "rank": rank}
            hits.append(hit)

        return hits

    def get_by_id(self, candidate_id: str) -> dict[str, Any] | None:
        """Fetch a single candidate's metadata by UUID."""
        result = self._collection.get(
            ids=[candidate_id],
            include=["metadatas", "documents"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        doc = result["documents"][0]
        return {**meta, "document": doc}

    def count(self) -> int:
        """Return the number of indexed candidates."""
        return self._collection.count()

    def delete_all(self) -> None:
        """
        Drop and recreate the collection (used when force re-ingesting).
        """
        settings = get_settings()
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection cleared and recreated.")
