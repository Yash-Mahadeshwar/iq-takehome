# Expert Network Search Copilot

An AI-powered backend API that lets you search for subject-matter experts through natural language queries. Built with FastAPI, ChromaDB, and OpenRouter.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Reference](#api-reference)
3. [How to Use (Step-by-Step)](#how-to-use-step-by-step)
4. [Example curl Requests](#example-curl-requests)
5. [Design Document](#design-document)
6. [Configuration Reference](#configuration-reference)
7. [Project Structure](#project-structure)

---

## Quick Start

### Prerequisites

- Python 3.9+
- Network access to the PostgreSQL source database
- An [OpenRouter](https://openrouter.ai) API key (provided)

### 1. Clone and set up

```bash
git clone <repo>
cd AgentAssist

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — the defaults already contain working credentials
```

The `.env` file needs these two values (already filled in from the credentials document):

```
POSTGRES_URL=postgresql://developer:devread2024@34.79.32.228:5432/candidate_profiles
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at:
- **Swagger UI** (interactive docs): http://localhost:8000/docs
- **ReDoc** (clean docs): http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

### 4. Ingest the data (first-time setup, ~3-5 minutes)

Before searching, you must build the vector index from the PostgreSQL database.

```bash
# Trigger ingestion
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'

# Poll for progress every few seconds until status == "completed"
curl http://localhost:8000/ingest/status
```

### 5. Search for experts

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find regulatory affairs experts with pharma experience in the Middle East"}'
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check — returns ChromaDB count, model info |
| `POST` | `/ingest` | Trigger ETL + embedding pipeline (async, returns 202) |
| `GET` | `/ingest/status` | Poll ingestion progress |
| `POST` | `/chat` | **Main endpoint** — natural language expert search |
| `GET` | `/experts/{id}` | Get full profile for a specific expert by UUID |
| `GET` | `/conversations` | List all active conversation sessions |
| `GET` | `/conversations/{id}` | Get conversation history + metadata |
| `DELETE` | `/conversations/{id}` | Delete a conversation |

### POST /chat — Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Natural language search query (3–2000 chars) |
| `conversation_id` | string | no | UUID from a previous `/chat` response, for follow-up queries |
| `top_k` | integer | no | Results to return (1–50, default: 10) |
| `explain` | boolean | no | Generate LLM explanations per match (default: true) |

### POST /chat — Response Fields

| Field | Description |
|-------|-------------|
| `conversation_id` | Save this to continue the conversation |
| `summary` | AI-generated conversational summary sentence |
| `intent` | The search intent as understood by the AI |
| `search_text` | The expanded query used for vector search (debug) |
| `total_found` | Total matching experts (may exceed `top_k`) |
| `results[]` | Ordered list of matching expert profiles |
| `results[].match_score` | Cosine similarity score 0–1 (higher = better) |
| `results[].explanation` | One-sentence LLM explanation of why they match |

---

## How to Use (Step-by-Step)

### Basic Search

1. Send a `POST /chat` with any natural language query.
2. The AI rewrites your query for richer semantic matching.
3. You receive ranked experts with match scores and explanations.
4. Save the `conversation_id` for follow-up queries.

### Conversational Follow-ups

Include the `conversation_id` from the previous response to refine results:

```
Turn 1: "Find regulatory affairs experts in pharma in the Middle East"
         → Returns 10 experts, conversation_id = "abc-123"

Turn 2: "Filter those to only people based in Saudi Arabia"
         (with conversation_id = "abc-123")
         → AI understands "those" refers to the previous results

Turn 3: "Which of them have more than 10 years of experience?"
         (with conversation_id = "abc-123")
         → Further refinement with full context
```

### Speed vs. Quality Trade-offs

- **Faster responses**: Set `"explain": false` to skip per-result LLM explanations (saves ~1s).
- **More results**: Increase `top_k` up to 50.
- **Better precision**: Write detailed queries — more context means better semantic matching.

---

## Example curl Requests

### Health check

```bash
curl http://localhost:8000/health
```

### Trigger ingestion

```bash
# Standard ingestion (safe to re-run — uses upsert)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'

# Force full rebuild (use when switching embedding models)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_rebuild": true}'
```

### Poll ingestion status

```bash
curl http://localhost:8000/ingest/status
```

Response while running:
```json
{
  "status": "running",
  "total": 10120,
  "processed": 3400,
  "failed": 0,
  "progress_pct": 33.6,
  "elapsed_seconds": 45.2,
  "error": null,
  "force_rebuild": false
}
```

### Search examples

```bash
# Regulatory affairs pharma in Middle East
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find me regulatory affairs experts with experience in the pharmaceutical industry in the Middle East",
    "top_k": 10
  }'

# Senior ML engineers in Europe
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Senior machine learning engineers with Python and PyTorch in Germany or Netherlands",
    "top_k": 5
  }'

# Finance professionals who speak Arabic
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment banking professionals who speak Arabic with CFA certification",
    "top_k": 8
  }'

# Fast search without explanations
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Supply chain managers with logistics experience in Asia",
    "top_k": 20,
    "explain": false
  }'

# Follow-up (refine previous results)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Filter those to only people based in Saudi Arabia",
    "conversation_id": "PASTE_CONVERSATION_ID_HERE",
    "top_k": 5
  }'
```

### Get expert profile by ID

```bash
# candidate_id comes from a /chat result
curl http://localhost:8000/experts/70222c8e-2b7a-4a9e-bc42-9ae3eaa2a89a
```

### Manage conversations

```bash
# List all active conversations
curl http://localhost:8000/conversations

# Get full message history for a conversation
curl http://localhost:8000/conversations/CONVERSATION_ID

# Delete a conversation
curl -X DELETE http://localhost:8000/conversations/CONVERSATION_ID
```

---

## Design Document

### Embedding Strategy

**One embedding per candidate** — each candidate's full profile is serialised into a single rich text document and encoded into one vector.

The document looks like this:

```
Name: Sara Ali
Headline: Regulatory Affairs Director | Pharmaceutical | GCC
Location: Riyadh, Saudi Arabia
Nationality: Lebanese
Years of Experience: 16

Work Experience:
- Regulatory Affairs Director at Novartis Saudi Arabia (Pharmaceutical industry) [Current]: Led GCC drug registration for 12 novel biologics...
- Senior RA Manager at Sanofi (Pharmaceutical industry): ...

Skills: Regulatory Affairs (Expert, 8 yrs) | Drug Registration (Expert, 10 yrs) | ICH Guidelines (Advanced) | ...

Education:
- MSc in Pharmaceutical Sciences from American University of Beirut (2009)

Languages: Arabic (Native), English (Fluent), French (Intermediate)
```

**Why one vector per candidate instead of chunking?**

Splitting into "skills chunk" + "experience chunk" would require multi-vector retrieval and a re-ranking step to merge and de-duplicate results across chunks — unnecessary complexity for profiles of 200–600 tokens (well within the model's 512-token window). A single holistic vector captures the full expertise surface area.

**Why include each field:**
- `headline` — the single richest sentence describing expertise
- `work_experience` with descriptions — domain knowledge, accomplishments, industry context
- `skills` with proficiency + years — key for skill-based queries
- `education` — important for academic and research roles
- `industries` — enables implicit industry filtering via semantics
- `location` — handles geo-specific queries even without hard filters
- `languages` — critical for Middle East / multilingual talent searches

### Vector Database: ChromaDB

**Why ChromaDB over alternatives?**

| Requirement | ChromaDB | Pinecone | Qdrant | pgvector | FAISS |
|---|---|---|---|---|---|
| Zero infrastructure | ✅ In-process | ❌ Cloud-only | ❌ Docker | ⚠️ Needs Postgres | ✅ |
| Persistent storage | ✅ File-based | ✅ | ✅ | ✅ | ❌ In-memory |
| Metadata filtering | ✅ `where` | ✅ | ✅ | ✅ SQL | ❌ |
| Cost | ✅ Free | ❌ $70+/mo | ✅ OSS | ✅ Free | ✅ Free |
| 10K–100K vectors | ✅ HNSW | ✅ | ✅ | ✅ | ✅ |

ChromaDB is the best fit: zero setup, persistent, free, with rich metadata filtering. Upsert semantics make re-ingestion idempotent.

### Embedding Model: `all-MiniLM-L6-v2`

A sentence-transformer with:
- **384 dimensions** — compact storage (10K vectors ≈ 15 MB on disk)
- **~2,000 sentences/sec on CPU** — ingests all 10K profiles in under 60 seconds
- **Trained on 1B+ sentence pairs** — excellent semantic similarity performance
- **No API key** — downloaded once (~90 MB), cached locally

To use a higher-quality model, set `EMBEDDING_MODEL=BAAI/bge-large-en-v1.5` in `.env` and re-ingest with `force_rebuild=true`.

### Query Handling Pipeline

```
User query
    │
    ▼
[1] QUERY REWRITING (Claude 3 Haiku via OpenRouter)
    - Expands abbreviations and adds domain synonyms
    - Extracts structured filters: country, industry, min_years_experience
    - Generates an "intent" summary sentence
    │
    ▼ expanded search_text
[2] EMBEDDING (sentence-transformers, local)
    - Encode search_text into 384-dim vector
    │
    ▼ query vector
[3] VECTOR SEARCH (ChromaDB cosine similarity)
    - Retrieve top_k × 3 candidates (over-fetch to allow filtering)
    │
    ▼ raw hits with cosine scores
[4] POST-FILTERING (Python)
    - Apply hard filters: country, industry, min_years_experience
    - Soft/substring matching handles name variants (UAE vs United Arab Emirates)
    - Trim to requested top_k
    │
    ▼ filtered ranked candidates
[5] EXPLANATION GENERATION (Claude 3 Haiku, batched)
    - One specific sentence per result explaining the match
    │
    ▼
[6] SUMMARY GENERATION (Claude 3 Haiku)
    - Conversational "I found X experts..." response sentence
    │
    ▼
JSON response
```

**Query rewriting is the key quality lever.** The raw query "pharma regulatory Middle East" is semantically sparse. After expansion: *"Regulatory affairs professional with pharmaceutical drug registration experience in the GCC region including Saudi Arabia, UAE, and Kuwait, with knowledge of ICH guidelines and regional health authority requirements."* This matches a much richer set of relevant profiles.

### Conversational Context

Conversations are stored in-memory as OpenAI-format `[{role, content}]` message lists. The last 6 messages (3 exchanges) are injected into the query rewriting LLM call. This lets the model resolve:

- *"those experts"* → the previous search results
- *"filter them to Saudi Arabia"* → add a location constraint
- *"which ones speak Arabic?"* → add a language criterion

Conversations expire after 60 minutes of inactivity (configurable).

### LLM Choice: `anthropic/claude-3-haiku`

Three tasks, all requiring structured output and fast latency:

1. **Query rewriting** → JSON with `search_text`, `filters`, `intent`
2. **Explanation generation** → JSON array of one-sentence strings
3. **Summary generation** → single conversational sentence

**Why Haiku?** 0.3–0.5s latency, ~$0.0001/query, strong JSON instruction following. Available on OpenRouter. Configurable via `OPENROUTER_MODEL` env var.

### Trade-offs

| Decision | Trade-off |
|---|---|
| Local embeddings (no API) | Slightly lower peak quality than `text-embedding-3-large`; zero cost, offline-capable |
| Single vector per candidate | Cannot isolate one specific skill area; holistic only |
| In-memory conversations | Lost on server restart; swap for Redis in production |
| Over-fetch × 3 for filtering | Small extra vector search cost; avoids extra DB roundtrip |
| HNSW approximate search | ~1% recall loss vs exact; imperceptible at 10K candidates |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_URL` | — | PostgreSQL DSN (required) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required) |
| `OPENROUTER_MODEL` | `anthropic/claude-3-haiku` | LLM model on OpenRouter |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage directory |
| `CHROMA_COLLECTION_NAME` | `expert_profiles` | Collection name |
| `INGEST_BATCH_SIZE` | `100` | Candidates per embedding batch |
| `DEFAULT_TOP_K` | `10` | Default results per search |
| `MAX_TOP_K` | `50` | Hard cap on results |
| `CONVERSATION_TTL_MINUTES` | `60` | Evict conversations after N minutes idle |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `ENVIRONMENT` | `development` | Deployment label |

---

## Project Structure

```
AgentAssist/
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .env                         # Your credentials (git-ignored)
├── chroma_data/                 # ChromaDB on-disk storage (created after /ingest)
│
└── app/
    ├── config.py                # Pydantic-settings configuration loader
    │
    ├── core/                    # Business logic
    │   ├── postgres.py          # PostgreSQL extraction + profile document builder
    │   ├── embedder.py          # sentence-transformers wrapper (local, cached)
    │   ├── vector_store.py      # ChromaDB wrapper (upsert, similarity search, get)
    │   ├── llm.py               # OpenRouter client (rewrite, explain, summarise)
    │   ├── ingestion.py         # ETL orchestrator with progress tracking
    │   ├── search.py            # Full pipeline: rewrite → embed → search → explain
    │   └── conversation.py      # In-memory conversation store with TTL eviction
    │
    ├── models/                  # Pydantic v2 schemas
    │   ├── requests.py          # IngestRequest, ChatRequest
    │   └── responses.py         # ChatResponse, ExpertResult, HealthResponse, etc.
    │
    └── routers/                 # FastAPI route handlers
        ├── health.py            # GET /health
        ├── ingest.py            # POST /ingest, GET /ingest/status
        ├── chat.py              # POST /chat
        └── experts.py           # GET /experts/{id}, /conversations CRUD
```

---

## Running in Production

```bash
# Gunicorn with uvicorn workers (recommended)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or plain uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

For multi-instance deployments:
- Mount `chroma_data/` on shared storage (NFS, EFS, etc.)
- Replace in-memory conversations with Redis (`redis-py` + TTL keys)
- Tighten CORS `allow_origins` in `main.py`
- Add authentication middleware (API key header or OAuth2)
