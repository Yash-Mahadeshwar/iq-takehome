"""
POST /chat — conversational expert search endpoint.

This is the primary user-facing endpoint. It accepts a natural language
query, runs the full search pipeline, and returns structured expert results
with LLM-generated explanations.

Conversation support
────────────────────
Pass the ``conversation_id`` returned from a previous call to continue a
conversation. The LLM uses the prior message history to resolve references
like "those experts", "filter them", "only people in Dubai", etc.

Error handling
──────────────
- 422 — invalid request body (Pydantic validation failure)
- 503 — ChromaDB is empty (ingestion has not been run)
- 500 — unexpected internal error
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.conversation import get_conversation_manager
from app.core.search import search_experts
from app.core.vector_store import VectorStore
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse, ExpertResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat / Search"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational expert search",
    description=(
        "**The core search endpoint.**\n\n"
        "Submit a natural language query to find matching subject-matter experts "
        "from the talent network. The AI assistant:\n\n"
        "1. Rewrites and expands your query for richer semantic matching.\n"
        "2. Searches the vector database for the most relevant expert profiles.\n"
        "3. Generates a one-sentence explanation for each match (if `explain=true`).\n"
        "4. Returns a conversational summary of what was found.\n\n"
        "**Follow-up queries**: include the `conversation_id` from a previous "
        "response to refine results — e.g. 'Filter those to only people in UAE'.\n\n"
        "**Prerequisites**: run `POST /ingest` at least once before searching."
    ),
)
def chat(body: ChatRequest) -> ChatResponse:
    # ── Guard: ensure the index is populated ─────────────────────────────────
    try:
        vs = VectorStore()
        if vs.count() == 0:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The expert index is empty. "
                    "Run POST /ingest first to populate the vector database."
                ),
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("VectorStore init failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

    # ── Conversation setup ────────────────────────────────────────────────────
    manager = get_conversation_manager()
    conv_id = manager.get_or_create(body.conversation_id)
    history = manager.get_messages(conv_id)

    # ── Run search pipeline ───────────────────────────────────────────────────
    try:
        result = search_experts(
            user_query=body.query,
            top_k=body.top_k,
            conversation_history=history,
            explain=body.explain,
        )
    except Exception as e:
        logger.exception("Search pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # ── Update conversation history ───────────────────────────────────────────
    manager.add_message(conv_id, "user", body.query)
    manager.set_last_results(conv_id, result["results"])
    assistant_reply = (
        f"{result['summary']} "
        f"(Showing {len(result['results'])} of {result['total_found']} matches)"
    )
    manager.add_message(conv_id, "assistant", assistant_reply)

    # ── Build response ────────────────────────────────────────────────────────
    expert_results = [ExpertResult(**r) for r in result["results"]]

    return ChatResponse(
        conversation_id=conv_id,
        summary=result["summary"],
        intent=result["intent"],
        search_text=result["search_text"],
        total_found=result["total_found"],
        results=expert_results,
    )
