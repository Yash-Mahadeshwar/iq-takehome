"""
Expert Network Search Copilot — FastAPI application entry point.

Start the server:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
OpenAPI JSON: http://localhost:8000/openapi.json
"""
from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, experts, health, ingest

# ─── Logging ──────────────────────────────────────────────────────────────────

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


settings = get_settings()
_configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

# ─── App factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Expert Network Search Copilot",
    description="""
## Overview
AI-powered expert talent search backed by a vector database.

### Workflow
1. **Ingest** (one-time): `POST /ingest` — extracts all candidates from PostgreSQL,
   generates semantic embeddings, and stores them in ChromaDB.
   Poll `GET /ingest/status` to monitor progress.

2. **Search**: `POST /chat` — submit a natural language query and receive
   ranked expert profiles with AI-generated relevance explanations.

3. **Follow-up**: include `conversation_id` from a previous `/chat` response
   to refine results conversationally (e.g. *"Filter those to only people in UAE"*).

### Key endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health check |
| `/ingest` | POST | Trigger ETL + embedding pipeline |
| `/ingest/status` | GET | Poll ingestion progress |
| `/chat` | POST | Conversational expert search |
| `/experts/{id}` | GET | Full profile for a specific expert |
| `/conversations` | GET | List active conversations |
| `/conversations/{id}` | GET | Get conversation history |
| `/conversations/{id}` | DELETE | Delete a conversation |
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ─── CORS (permissive for development; tighten in production) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(experts.router)


# ─── Root redirect ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "message": "Expert Network Search Copilot",
        "docs": "/docs",
        "health": "/health",
    }


# ─── Startup log ─────────────────────────────────────────────────────────────
logger.info(
    "Expert Network Search Copilot starting — model=%s  env=%s",
    settings.openrouter_model,
    settings.environment,
)
