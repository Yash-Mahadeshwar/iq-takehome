"""
PostgreSQL data extraction layer.

Connects to the candidate profiles database and returns fully-enriched
candidate records by JOINing across all related tables (work experience,
education, skills, languages, etc.).

All queries are read-only SELECT statements.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extras

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Connection ───────────────────────────────────────────────────────────────

@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager that yields a database connection and closes it on exit."""
    settings = get_settings()
    conn = psycopg2.connect(settings.postgres_url, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


# ─── Queries ──────────────────────────────────────────────────────────────────

_CANDIDATES_BASE_SQL = """
SELECT
    c.id                                                AS candidate_id,
    c.first_name,
    c.last_name,
    CONCAT(c.first_name, ' ', c.last_name)              AS full_name,
    c.email,
    c.phone,
    c.headline,
    c.years_of_experience,
    c.gender,
    c.date_of_birth,
    c.created_at,
    ci.name                                             AS city,
    co.name                                             AS country,
    co.code                                             AS country_code,
    nat.name                                            AS nationality
FROM candidates c
LEFT JOIN cities    ci  ON c.city_id         = ci.id
LEFT JOIN countries co  ON ci.country_id     = co.id
LEFT JOIN countries nat ON c.nationality_id  = nat.id
"""


def fetch_all_candidates(batch_size: int = 500) -> list[dict[str, Any]]:
    """
    Return all candidate base records (no sub-tables yet).
    Used by the ingestion pipeline as the outer loop.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CANDIDATES_BASE_SQL + " ORDER BY c.created_at")
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def fetch_candidate_by_id(candidate_id: str) -> dict[str, Any] | None:
    """Fetch a single candidate's base record by UUID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                _CANDIDATES_BASE_SQL + " WHERE c.id = %s",
                (candidate_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def fetch_work_experience(candidate_ids: list[str]) -> dict[str, list[dict]]:
    """
    Return work experience records keyed by candidate_id.
    Ordered most-recent-first so index 0 is the current/latest role.
    """
    if not candidate_ids:
        return {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    we.candidate_id,
                    we.job_title,
                    co.name     AS company,
                    co.industry AS industry,
                    we.start_date,
                    we.end_date,
                    we.is_current,
                    we.description
                FROM work_experience we
                JOIN companies co ON we.company_id = co.id
                WHERE we.candidate_id = ANY(%s::uuid[])
                ORDER BY we.is_current DESC, we.start_date DESC NULLS LAST
                """,
                (candidate_ids,),
            )
            rows = cur.fetchall()

    result: dict[str, list[dict]] = {}
    for r in rows:
        cid = str(r["candidate_id"])
        result.setdefault(cid, []).append(dict(r))
    return result


def fetch_skills(candidate_ids: list[str]) -> dict[str, list[dict]]:
    """Return skills keyed by candidate_id, ordered by years_of_experience desc."""
    if not candidate_ids:
        return {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    cs.candidate_id,
                    s.name                  AS skill,
                    sc.name                 AS category,
                    cs.proficiency_level,
                    cs.years_of_experience  AS skill_years
                FROM candidate_skills cs
                JOIN skills          s  ON cs.skill_id      = s.id
                LEFT JOIN skill_categories sc ON s.category_id = sc.id
                WHERE cs.candidate_id = ANY(%s::uuid[])
                ORDER BY cs.years_of_experience DESC NULLS LAST
                """,
                (candidate_ids,),
            )
            rows = cur.fetchall()

    result: dict[str, list[dict]] = {}
    for r in rows:
        cid = str(r["candidate_id"])
        result.setdefault(cid, []).append(dict(r))
    return result


def fetch_education(candidate_ids: list[str]) -> dict[str, list[dict]]:
    """Return education records keyed by candidate_id, ordered by graduation_year desc."""
    if not candidate_ids:
        return {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.candidate_id,
                    i.name   AS institution,
                    d.name   AS degree,
                    f.name   AS field_of_study,
                    e.graduation_year,
                    e.grade
                FROM education e
                JOIN institutions    i ON e.institution_id    = i.id
                JOIN degrees         d ON e.degree_id         = d.id
                JOIN fields_of_study f ON e.field_of_study_id = f.id
                WHERE e.candidate_id = ANY(%s::uuid[])
                ORDER BY e.graduation_year DESC NULLS LAST
                """,
                (candidate_ids,),
            )
            rows = cur.fetchall()

    result: dict[str, list[dict]] = {}
    for r in rows:
        cid = str(r["candidate_id"])
        result.setdefault(cid, []).append(dict(r))
    return result


def fetch_languages(candidate_ids: list[str]) -> dict[str, list[dict]]:
    """Return languages keyed by candidate_id."""
    if not candidate_ids:
        return {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    cl.candidate_id,
                    l.name  AS language,
                    pl.name AS proficiency
                FROM candidate_languages cl
                JOIN languages         l  ON cl.language_id          = l.id
                JOIN proficiency_levels pl ON cl.proficiency_level_id = pl.id
                WHERE cl.candidate_id = ANY(%s::uuid[])
                ORDER BY pl.rank DESC NULLS LAST
                """,
                (candidate_ids,),
            )
            rows = cur.fetchall()

    result: dict[str, list[dict]] = {}
    for r in rows:
        cid = str(r["candidate_id"])
        result.setdefault(cid, []).append(dict(r))
    return result


def fetch_total_candidate_count() -> int:
    """Return total number of candidates in the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM candidates")
            row = cur.fetchone()
            return int(row["n"])


# ─── Profile builder ──────────────────────────────────────────────────────────

def build_full_profiles(
    candidates: list[dict],
    work_exp_map: dict[str, list[dict]],
    skills_map: dict[str, list[dict]],
    education_map: dict[str, list[dict]],
    languages_map: dict[str, list[dict]],
) -> list[dict[str, Any]]:
    """
    Merge base candidate records with all sub-tables into rich profile dicts.
    Also constructs the ``profile_text`` field used for embedding generation.
    """
    profiles = []
    for cand in candidates:
        cid = str(cand["candidate_id"])

        work = work_exp_map.get(cid, [])
        skills = skills_map.get(cid, [])
        edu = education_map.get(cid, [])
        langs = languages_map.get(cid, [])

        # Derive current / most-recent role
        current_role = next((w for w in work if w.get("is_current")), work[0] if work else None)

        # Unique industries across all work experience
        industries = list({w["industry"] for w in work if w.get("industry")})

        # Skill names only (for faceted filtering)
        skill_names = [s["skill"] for s in skills]
        skill_categories = list({s["category"] for s in skills if s.get("category")})

        profile = {
            **cand,
            "work_experience": work,
            "skills": skills,
            "education": edu,
            "languages": langs,
            "current_title": current_role["job_title"] if current_role else None,
            "current_company": current_role["company"] if current_role else None,
            "industries": industries,
            "skill_names": skill_names,
            "skill_categories": skill_categories,
            "profile_text": _build_profile_text(cand, work, skills, edu, langs),
        }
        profiles.append(profile)
    return profiles


def _build_profile_text(
    cand: dict,
    work: list[dict],
    skills: list[dict],
    edu: list[dict],
    langs: list[dict],
) -> str:
    """
    Construct the free-text representation of a candidate used for embedding.

    Design rationale:
    - All semantically important fields are included so the embedding captures
      the full expertise surface area.
    - Fields are labelled (e.g. "Skills:") so the model can learn field
      semantics from co-occurrence patterns.
    - Work experience descriptions are included verbatim for rich context.
    - The order roughly mirrors a LinkedIn-style profile so the language model
      used for embeddings (trained on web text) can leverage its priors.
    """
    parts: list[str] = []

    # Identity & headline
    name = f"{cand.get('first_name', '')} {cand.get('last_name', '')}".strip()
    parts.append(f"Name: {name}")
    if cand.get("headline"):
        parts.append(f"Headline: {cand['headline']}")

    # Location
    location_parts = [p for p in [cand.get("city"), cand.get("country")] if p]
    if location_parts:
        parts.append(f"Location: {', '.join(location_parts)}")
    if cand.get("nationality"):
        parts.append(f"Nationality: {cand['nationality']}")

    # Experience summary
    yoe = cand.get("years_of_experience")
    if yoe is not None:
        parts.append(f"Years of Experience: {yoe}")

    # Work experience
    if work:
        work_lines = []
        for w in work:
            role_parts = [w["job_title"]]
            if w.get("company"):
                role_parts.append(f"at {w['company']}")
            if w.get("industry"):
                role_parts.append(f"({w['industry']} industry)")
            if w.get("is_current"):
                role_parts.append("[Current]")
            line = " ".join(role_parts)
            if w.get("description"):
                line += f": {w['description']}"
            work_lines.append(line)
        parts.append("Work Experience:\n" + "\n".join(f"- {l}" for l in work_lines))

    # Skills
    if skills:
        skill_tokens = []
        for s in skills:
            token = s["skill"]
            if s.get("proficiency_level"):
                token += f" ({s['proficiency_level']}"
                if s.get("skill_years"):
                    token += f", {s['skill_years']} yrs"
                token += ")"
            skill_tokens.append(token)
        parts.append("Skills: " + " | ".join(skill_tokens))

    # Education
    if edu:
        edu_lines = []
        for e in edu:
            line_parts = [e["degree"], "in", e["field_of_study"], "from", e["institution"]]
            if e.get("graduation_year"):
                line_parts.append(f"({e['graduation_year']})")
            edu_lines.append(" ".join(line_parts))
        parts.append("Education:\n" + "\n".join(f"- {l}" for l in edu_lines))

    # Languages
    if langs:
        lang_tokens = [
            f"{l['language']} ({l['proficiency']})" if l.get("proficiency") else l["language"]
            for l in langs
        ]
        parts.append("Languages: " + ", ".join(lang_tokens))

    return "\n\n".join(parts)
