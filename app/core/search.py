"""
Expert search engine.

Query handling pipeline
────────────────────────
1. **Query rewriting** (LLM):
   The raw user query is passed to Claude which expands it into richer semantic
   text and extracts any structured filters (country, industry, min_experience).
   Example:
     Input:  "regulatory affairs pharma Middle East"
     Output: "Senior regulatory affairs specialist with pharmaceutical drug
              approval experience in Saudi Arabia, UAE or Gulf region.
              Knowledge of GCC regulatory frameworks, ICH guidelines."

2. **Embedding** (sentence-transformers):
   The expanded ``search_text`` is embedded using the same model used during
   ingestion, ensuring the query lives in the same vector space.

3. **Vector search** (ChromaDB):
   Retrieve top-K candidates by cosine similarity. We over-fetch
   (``top_k * 2``) to allow for post-filtering without hitting the DB again.

4. **Post-filtering** (Python):
   Apply hard filters extracted from the query (e.g. country == "Saudi Arabia")
   to narrow the candidate set.

5. **Explanation generation** (LLM):
   For the final top-N results, Claude generates a one-sentence "why this
   person matches" explanation personalised to the user's search intent.

6. **Conversational summary** (LLM):
   A brief conversational sentence ("I found 6 pharma experts in the Gulf…")
   is generated as the top-level response message.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.core.embedder import get_embedder
from app.core.llm import explain_matches, generate_summary_response, rewrite_query
from app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Module-level singleton (initialised lazily on first call)
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


# ─── Main search function ─────────────────────────────────────────────────────

def search_experts(
    user_query: str,
    top_k: int | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    explain: bool = True,
) -> dict[str, Any]:
    """
    Run the full expert search pipeline and return a structured result dict.

    Args:
        user_query:           Natural language search query from the user.
        top_k:                How many experts to return (capped at max_top_k).
        conversation_history: Prior conversation turns for follow-up resolution.
        explain:              If True, generate LLM explanations for top results.

    Returns:
        {
            "intent":        str,
            "search_text":   str,       # expanded query used for embedding
            "total_found":   int,
            "results":       list[ExpertResult],
            "summary":       str,       # conversational sentence
        }
    """
    settings = get_settings()
    effective_top_k = min(
        top_k or settings.default_top_k,
        settings.max_top_k,
    )

    # ── 1. Query rewriting ────────────────────────────────────────────────────
    logger.info("Rewriting query: %r", user_query[:80])
    rewrite = rewrite_query(user_query, conversation_history)
    search_text: str = rewrite.get("search_text", user_query)
    intent: str = rewrite.get("intent", user_query)
    filters: dict = rewrite.get("filters", {}) or {}
    logger.info("Rewritten → intent=%r  filters=%s", intent[:80], filters)

    # ── 2. Embed the expanded query ───────────────────────────────────────────
    embedder = get_embedder()
    query_vec = embedder.embed_one(search_text)

    # ── 3. Vector search (over-fetch to allow filtering) ─────────────────────
    vs = _get_vector_store()
    # Fetch a large pool; we'll filter down from it
    fetch_k = min(max(effective_top_k * 5, 100), settings.max_top_k * 5, vs.count())
    raw_hits = vs.similarity_search(query_embedding=query_vec, top_k=fetch_k)

    # ── 4. Post-filter (with graceful fallback) ───────────────────────────────
    filtered = _apply_filters(raw_hits, filters)

    # If hard filters eliminate everything, fall back to unfiltered results
    # so users always get a useful response
    if not filtered and raw_hits:
        logger.info(
            "Hard filters produced 0 results — falling back to unfiltered top-%d.",
            effective_top_k,
        )
        filtered = raw_hits

    # Trim to requested top_k
    final_hits = filtered[:effective_top_k]
    total_found = len(filtered)

    # ── 5. Explanation generation ─────────────────────────────────────────────
    explanations: list[str] = []
    if explain and final_hits:
        try:
            explanations = explain_matches(intent, final_hits, top_n=len(final_hits))
        except Exception as e:
            logger.warning("Explanation generation failed: %s", e)
            explanations = [""] * len(final_hits)

    # ── 6. Conversational summary ─────────────────────────────────────────────
    summary = ""
    try:
        summary = generate_summary_response(user_query, intent, total_found, final_hits)
    except Exception as e:
        logger.warning("Summary generation failed: %s", e)
        summary = f"Found {total_found} matching experts."

    # ── 7. Build result dicts ─────────────────────────────────────────────────
    results = []
    for i, hit in enumerate(final_hits):
        exp = explanations[i] if i < len(explanations) else ""
        results.append(_hit_to_result(hit, explanation=exp))

    return {
        "intent": intent,
        "search_text": search_text,
        "total_found": total_found,
        "results": results,
        "summary": summary,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────


# ─── Region → countries mapping ──────────────────────────────────────────────
# Expands geographic region names into lists of individual country names so
# "Middle East" correctly filters candidates in Saudi Arabia, UAE, etc.

_REGION_MAP: dict[str, list[str]] = {
    "middle east": [
        "saudi arabia", "united arab emirates", "uae", "egypt", "qatar",
        "kuwait", "bahrain", "oman", "jordan", "lebanon", "israel",
        "iraq", "iran", "yemen", "syria", "palestine",
    ],
    "gulf": [
        "saudi arabia", "united arab emirates", "uae", "qatar",
        "kuwait", "bahrain", "oman",
    ],
    "gcc": [
        "saudi arabia", "united arab emirates", "uae", "qatar",
        "kuwait", "bahrain", "oman",
    ],
    "europe": [
        "germany", "france", "united kingdom", "uk", "spain", "italy",
        "netherlands", "belgium", "switzerland", "sweden", "norway",
        "denmark", "finland", "austria", "poland", "portugal", "greece",
        "czech republic", "hungary", "romania", "ukraine",
    ],
    "north america": ["united states", "usa", "canada", "mexico"],
    "latin america": [
        "brazil", "argentina", "colombia", "chile", "peru", "venezuela",
        "ecuador", "uruguay", "bolivia", "paraguay",
    ],
    "south asia": ["india", "pakistan", "bangladesh", "sri lanka", "nepal"],
    "southeast asia": [
        "singapore", "malaysia", "indonesia", "thailand", "vietnam",
        "philippines", "myanmar",
    ],
    "east asia": ["china", "japan", "south korea", "taiwan", "hong kong"],
    "africa": [
        "nigeria", "south africa", "kenya", "ethiopia", "ghana", "egypt",
        "morocco", "tanzania", "uganda", "algeria", "tunisia",
    ],
}

# Industry synonym groups — expands narrow terms to broader categories
_INDUSTRY_SYNONYMS: dict[str, list[str]] = {
    "pharmaceutical": [
        "pharmaceutical", "pharma", "biotech", "biotechnology",
        "hospitals and health care", "health care", "medical",
        "life sciences", "drug", "healthcare",
    ],
    "pharma": [
        "pharmaceutical", "pharma", "biotech", "biotechnology",
        "hospitals and health care",
    ],
    "finance": [
        "finance", "financial services", "banking", "investment",
        "insurance", "capital markets",
    ],
    "financial services": ["financial services", "finance", "banking", "insurance"],
    "tech": [
        "software development", "technology", "information and internet",
        "it services", "it consulting",
    ],
    "technology": [
        "software development", "technology", "information and internet",
        "it services", "it consulting",
    ],
    "healthcare": [
        "hospitals and health care", "health care", "medical", "pharma",
        "pharmaceutical", "biotechnology",
    ],
}


def _expand_geo(term: str) -> list[str]:
    """Return the term itself plus any region expansion countries."""
    t = term.lower()
    return [t] + _REGION_MAP.get(t, [])


def _expand_industry(term: str) -> list[str]:
    """Return the term itself plus any industry synonyms."""
    t = term.lower()
    return [t] + _INDUSTRY_SYNONYMS.get(t, [])


def _apply_filters(hits: list[dict], filters: dict) -> list[dict]:
    """
    Apply soft post-filters extracted from the user query.

    Features:
    - Case-insensitive substring matching.
    - Region expansion: "Middle East" → [Saudi Arabia, UAE, Egypt, …].
    - Industry synonym expansion: "pharmaceutical" → [pharma, biotech, …].
    - Each filter applied independently; only non-empty filter values are used.
    """
    if not filters:
        return hits

    result = hits

    # ── Geographic filter ─────────────────────────────────────────────────────
    country_raw = (filters.get("country") or "").strip().lower()
    if country_raw:
        geo_terms = _expand_geo(country_raw)

        def _geo_match(h: dict) -> bool:
            candidate_country = (h.get("country") or "").lower()
            candidate_nat = (h.get("nationality") or "").lower()
            return any(
                t in candidate_country or t in candidate_nat
                for t in geo_terms
            )

        geo_filtered = [h for h in result if _geo_match(h)]
        # Only apply if it yields results (avoid empty result due to wrong LLM extraction)
        if geo_filtered:
            result = geo_filtered
        else:
            logger.debug("Geo filter '%s' matched 0 — skipping.", country_raw)

    # ── Industry filter ───────────────────────────────────────────────────────
    industry_raw = (filters.get("industry") or "").strip().lower()
    if industry_raw:
        ind_terms = _expand_industry(industry_raw)

        def _ind_match(h: dict) -> bool:
            ind_str = (h.get("industries") or "").lower()
            return any(t in ind_str for t in ind_terms)

        ind_filtered = [h for h in result if _ind_match(h)]
        # Only apply if it yields results
        if ind_filtered:
            result = ind_filtered
        else:
            logger.debug("Industry filter '%s' matched 0 — skipping.", industry_raw)

    # ── Min years of experience filter ────────────────────────────────────────
    min_yoe = filters.get("min_years_experience")
    if min_yoe is not None:
        try:
            min_yoe = int(min_yoe)
            yoe_filtered = [
                h for h in result
                if int(h.get("years_of_experience") or 0) >= min_yoe
            ]
            if yoe_filtered:
                result = yoe_filtered
        except (TypeError, ValueError):
            pass

    return result


def _hit_to_result(hit: dict[str, Any], explanation: str = "") -> dict[str, Any]:
    """
    Convert a raw ChromaDB hit into a clean result dict suitable for
    serialisation into the API response.
    """
    skill_names = [s.strip() for s in (hit.get("skill_names") or "").split(",") if s.strip()]
    industries = [i.strip() for i in (hit.get("industries") or "").split(",") if i.strip()]
    languages = [l.strip() for l in (hit.get("languages") or "").split(",") if l.strip()]

    return {
        "candidate_id": hit.get("candidate_id", ""),
        "full_name": hit.get("full_name", ""),
        "email": hit.get("email", ""),
        "headline": hit.get("headline", ""),
        "current_title": hit.get("current_title", ""),
        "current_company": hit.get("current_company", ""),
        "city": hit.get("city", ""),
        "country": hit.get("country", ""),
        "nationality": hit.get("nationality", ""),
        "years_of_experience": int(hit.get("years_of_experience") or 0),
        "industries": industries,
        "skill_names": skill_names,
        "education_summary": hit.get("education_summary", ""),
        "languages": languages,
        "match_score": float(hit.get("score", 0.0)),
        "rank": int(hit.get("rank", 0)),
        "explanation": explanation,
    }
