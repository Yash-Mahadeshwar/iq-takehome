"""
/experts — additional expert profile endpoints.

GET /experts/{candidate_id}  — fetch a single expert's full profile from ChromaDB
GET /conversations           — list active conversation sessions
GET /conversations/{id}      — get conversation detail + message history
DELETE /conversations/{id}   — delete a conversation
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.conversation import get_conversation_manager
from app.core.vector_store import VectorStore
from app.models.responses import (
    ConversationDetailResponse,
    ConversationInfo,
    DeleteResponse,
    ExpertDetailResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Experts & Conversations"])


# ─── Expert profile ───────────────────────────────────────────────────────────

@router.get(
    "/experts/{candidate_id}",
    response_model=ExpertDetailResponse,
    summary="Get expert profile by ID",
    description=(
        "Retrieve the full indexed profile for a specific expert by their UUID. "
        "The `candidate_id` is returned in every `/chat` response result. "
        "Returns 404 if the expert is not in the vector index (not yet ingested)."
    ),
)
def get_expert(candidate_id: str) -> ExpertDetailResponse:
    try:
        vs = VectorStore()
        doc = vs.get_by_id(candidate_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Expert '{candidate_id}' not found in the vector index.",
        )

    skill_names = [s.strip() for s in (doc.get("skill_names") or "").split(",") if s.strip()]
    industries = [i.strip() for i in (doc.get("industries") or "").split(",") if i.strip()]
    languages = [l.strip() for l in (doc.get("languages") or "").split(",") if l.strip()]

    return ExpertDetailResponse(
        candidate_id=doc.get("candidate_id", candidate_id),
        full_name=doc.get("full_name", ""),
        email=doc.get("email", ""),
        headline=doc.get("headline", ""),
        current_title=doc.get("current_title", ""),
        current_company=doc.get("current_company", ""),
        city=doc.get("city", ""),
        country=doc.get("country", ""),
        nationality=doc.get("nationality", ""),
        years_of_experience=int(doc.get("years_of_experience") or 0),
        industries=industries,
        skill_names=skill_names,
        education_summary=doc.get("education_summary", ""),
        languages=languages,
        profile_text=doc.get("document", doc.get("profile_text", "")),
    )


# ─── Conversations ────────────────────────────────────────────────────────────

@router.get(
    "/conversations",
    response_model=list[ConversationInfo],
    summary="List active conversations",
    description="Returns a list of all active (non-evicted) conversation sessions.",
)
def list_conversations() -> list[ConversationInfo]:
    manager = get_conversation_manager()
    return [ConversationInfo(**c) for c in manager.list_conversations()]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get conversation detail",
    description="Returns the full message history and metadata for a conversation.",
)
def get_conversation(conversation_id: str) -> ConversationDetailResponse:
    manager = get_conversation_manager()
    info = manager.get_conversation_info(conversation_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found or has expired.",
        )
    messages = manager.get_messages(conversation_id)
    return ConversationDetailResponse(**info, messages=messages)


@router.delete(
    "/conversations/{conversation_id}",
    response_model=DeleteResponse,
    summary="Delete a conversation",
    description="Permanently deletes a conversation and its history.",
)
def delete_conversation(conversation_id: str) -> DeleteResponse:
    manager = get_conversation_manager()
    deleted = manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return DeleteResponse(message=f"Conversation '{conversation_id}' deleted.")
