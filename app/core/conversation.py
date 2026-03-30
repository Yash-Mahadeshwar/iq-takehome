"""
Conversation manager.

Maintains per-session chat history so follow-up queries can reference
prior results (e.g. "Filter those to people based in Saudi Arabia").

Implementation
──────────────
- **In-memory dict** keyed by ``conversation_id`` (UUID string).
- Each conversation stores:
    - ``messages``   : list of {role, content} dicts (OpenAI format)
    - ``last_results``: list of expert result dicts from the last search
    - ``last_active`` : Unix timestamp for TTL eviction
- A background sweep removes conversations idle > ``CONVERSATION_TTL_MINUTES``.

Trade-offs
──────────
- In-memory means conversations are lost on server restart.
- For production: back with Redis (use redis-py + TTL keys) or a simple
  PostgreSQL table with a ``last_active`` column.
- The current approach is appropriate for a single-instance deployment with
  stateless scaling managed externally.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Thread-safe in-memory conversation store.

    Usage
    ─────
    manager = ConversationManager()

    # Start or resume a conversation
    conv_id = manager.get_or_create("existing-id-or-None")

    # Append a user turn
    manager.add_message(conv_id, "user", "Find pharma experts in UAE")

    # Retrieve history for LLM context injection
    history = manager.get_messages(conv_id)   # [{role, content}, …]

    # Store last search results (for follow-up resolution)
    manager.set_last_results(conv_id, expert_list)

    # Append the assistant's response
    manager.add_message(conv_id, "assistant", "I found 8 experts…")
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._start_eviction_thread()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def get_or_create(self, conversation_id: str | None) -> str:
        """
        Return an existing conversation_id if valid, otherwise create a new one.
        Calling this also updates ``last_active``.
        """
        with self._lock:
            if conversation_id and conversation_id in self._store:
                self._store[conversation_id]["last_active"] = time.time()
                return conversation_id
            new_id = str(uuid.uuid4())
            self._store[new_id] = {
                "messages": [],
                "last_results": [],
                "last_active": time.time(),
                "created_at": time.time(),
            }
            logger.debug("Created conversation %s", new_id)
            return new_id

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        with self._lock:
            conv = self._store.get(conversation_id)
            if conv is None:
                return
            conv["messages"].append({"role": role, "content": content})
            conv["last_active"] = time.time()

    def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
        """Return the full message history for LLM injection."""
        with self._lock:
            conv = self._store.get(conversation_id, {})
            return list(conv.get("messages", []))

    def set_last_results(
        self, conversation_id: str, results: list[dict[str, Any]]
    ) -> None:
        """Store the most recent search results for context in follow-ups."""
        with self._lock:
            conv = self._store.get(conversation_id)
            if conv is not None:
                conv["last_results"] = results
                conv["last_active"] = time.time()

    def get_last_results(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conv = self._store.get(conversation_id, {})
            return list(conv.get("last_results", []))

    def get_conversation_info(self, conversation_id: str) -> dict[str, Any] | None:
        """Return metadata about a conversation (for GET /conversations/{id})."""
        with self._lock:
            conv = self._store.get(conversation_id)
            if conv is None:
                return None
            return {
                "conversation_id": conversation_id,
                "message_count": len(conv["messages"]),
                "created_at": conv["created_at"],
                "last_active": conv["last_active"],
                "has_results": bool(conv["last_results"]),
            }

    def list_conversations(self) -> list[dict[str, Any]]:
        """Return brief info about all active conversations."""
        with self._lock:
            return [
                {
                    "conversation_id": cid,
                    "message_count": len(v["messages"]),
                    "last_active": v["last_active"],
                }
                for cid, v in self._store.items()
            ]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        with self._lock:
            return self._store.pop(conversation_id, None) is not None

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _evict_stale(self) -> None:
        ttl = get_settings().conversation_ttl_minutes * 60
        now = time.time()
        with self._lock:
            stale = [
                cid for cid, v in self._store.items()
                if now - v["last_active"] > ttl
            ]
            for cid in stale:
                del self._store[cid]
            if stale:
                logger.debug("Evicted %d stale conversations.", len(stale))

    def _start_eviction_thread(self) -> None:
        """Run eviction every 10 minutes in a daemon thread."""
        def _loop() -> None:
            while True:
                time.sleep(600)
                try:
                    self._evict_stale()
                except Exception as e:
                    logger.warning("Eviction error: %s", e)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()


# ── Application-scoped singleton ──────────────────────────────────────────────

_manager: ConversationManager | None = None


def get_conversation_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager
