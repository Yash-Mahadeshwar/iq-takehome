"""
OpenRouter LLM client.

OpenRouter exposes an OpenAI-compatible REST API, so we use the official
``openai`` Python SDK pointed at https://openrouter.ai/api/v1.

Model choice: ``anthropic/claude-3-haiku``
──────────────────────────────────────────
We use Claude 3 Haiku as the default for three reasons:
1. **Speed**: Sub-second latency for the small prompts we send (query
   rewriting, relevance explanations).
2. **Cost**: ~$0.0001 / query — negligible even at scale.
3. **Quality**: Strong instruction following, good at extracting structured
   intent from natural language.

Alternative models available on OpenRouter:
- ``openai/gpt-4o-mini``            — comparable cost, slightly different
                                       phrasing style.
- ``google/gemini-flash-1.5``       — very fast, multilingual strength.
- ``meta-llama/llama-3.1-8b-instruct`` — open-source, lower cost.

All LLM calls are synchronous so they can be used both in FastAPI async
route handlers (via ``run_in_executor``) and in CLI scripts.
We use tenacity for automatic retry on transient errors.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/expert-network-search",
            "X-Title": "Expert Network Search Copilot",
        },
    )


# ─── Core completion helper ───────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _complete(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Call the OpenRouter chat completions endpoint and return the text content.
    Retries up to 3 times on transient failures (rate limits, timeouts).
    """
    settings = get_settings()
    effective_model = model or settings.openrouter_model
    client = _get_client()

    response = client.chat.completions.create(
        model=effective_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    logger.debug("LLM response (%d chars): %s…", len(content), content[:120])
    return content.strip()


# ─── Task-specific functions ──────────────────────────────────────────────────

def rewrite_query(
    user_query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Transform the user's natural language query into a richer search query.

    Returns a dict with:
        ``search_text``  — expanded text ready for embedding
        ``filters``      — optional structured filters (country, industry, etc.)
        ``intent``       — one-sentence summary of what the user is looking for
    """
    system = (
        "You are a search query analyser for an expert talent network. "
        "Given a user query, extract the search intent and expand it for "
        "semantic vector search. Respond with JSON only, no markdown fences.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "search_text": "<expanded semantic search string>",\n'
        '  "filters": {"country": "<or null>", "industry": "<or null>", '
        '"min_years_experience": <int or null>},\n'
        '  "intent": "<one-sentence summary of what the user wants>"\n'
        "}\n\n"
        "For search_text: expand abbreviations, add synonyms, include relevant "
        "domain terms. Make it 2-4 sentences of rich descriptive text that a "
        "matching expert's profile would contain.\n"
        "For filters: only fill in values you are confident about from the query; "
        "otherwise use null."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    # Inject conversation history so follow-up queries are resolved correctly
    if conversation_history:
        for turn in conversation_history[-6:]:  # last 3 exchanges
            messages.append(turn)

    messages.append({"role": "user", "content": user_query})

    raw = _complete(messages, temperature=0.1, max_tokens=512)

    # Robust JSON parsing — handle models that wrap JSON in markdown fences
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON for query rewrite: %s", raw[:200])
        parsed = {
            "search_text": user_query,
            "filters": {},
            "intent": user_query,
        }

    return parsed


def explain_matches(
    user_intent: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
) -> list[str]:
    """
    For each of the top ``top_n`` candidates, generate a one-to-two sentence
    explanation of why they match the user's intent.

    Returns a list of explanation strings (same length as ``candidates[:top_n]``).
    """
    if not candidates:
        return []

    snippets = []
    for i, c in enumerate(candidates[:top_n]):
        snippet = (
            f"Candidate {i+1}: {c.get('full_name', 'N/A')}\n"
            f"  Title: {c.get('current_title', 'N/A')} at {c.get('current_company', 'N/A')}\n"
            f"  Location: {c.get('city', '')}, {c.get('country', '')}\n"
            f"  Experience: {c.get('years_of_experience', 'N/A')} years\n"
            f"  Industries: {c.get('industries', 'N/A')}\n"
            f"  Skills: {c.get('skill_names', 'N/A')}\n"
            f"  Education: {c.get('education_summary', 'N/A')}"
        )
        snippets.append(snippet)

    system = (
        "You are a talent search assistant. For each candidate snippet below, "
        "write exactly ONE sentence (max 40 words) explaining why they are a "
        "strong match for the search intent. Be specific — mention actual skills, "
        "industries, or locations from their profile. No bullet points.\n"
        "Respond with a JSON array of strings, one per candidate, in order."
    )
    user_msg = (
        f"Search intent: {user_intent}\n\n"
        + "\n\n".join(snippets)
    )

    raw = _complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.3,
        max_tokens=800,
    )

    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()

    try:
        explanations = json.loads(raw)
        if not isinstance(explanations, list):
            raise ValueError("expected list")
    except (json.JSONDecodeError, ValueError):
        logger.warning("LLM returned non-JSON for explanations: %s", raw[:200])
        explanations = [f"Matches your search for: {user_intent}"] * len(candidates[:top_n])

    # Pad or trim to match number of candidates
    result = list(explanations[:top_n])
    while len(result) < len(candidates[:top_n]):
        result.append(f"Relevant match for: {user_intent}")
    return result


def generate_summary_response(
    user_query: str,
    user_intent: str,
    result_count: int,
    top_candidates: list[dict[str, Any]],
) -> str:
    """
    Generate a short conversational response summarising the search results
    (e.g. "I found 8 regulatory affairs experts in the Middle East…").
    """
    names = ", ".join(c.get("full_name", "?") for c in top_candidates[:3])
    system = (
        "You are a helpful talent search assistant. Write a concise 1-2 sentence "
        "conversational response summarising the search results. Mention the "
        "result count and the type of experts found. Be friendly but professional."
    )
    user_msg = (
        f"User query: {user_query}\n"
        f"Intent: {user_intent}\n"
        f"Total results found: {result_count}\n"
        f"Top matches include: {names}"
    )
    return _complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.4,
        max_tokens=150,
    )
