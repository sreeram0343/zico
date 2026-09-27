"""
Conversational Session Memory Service for ZICO.

Provides:
    - Redis-backed short-term conversation context for multi-turn travel interactions.
    - Explicit 24-hour expiration policy to prevent unbounded memory growth.
    - In-memory fallback when Redis is offline.
    - Graceful degradation with zero unhandled exceptions.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from app.db.redis import get_redis

logger = logging.getLogger(__name__)

# Fallback local memory storage when Redis is unavailable
_LOCAL_SESSION_MEMORY: Dict[str, List[Dict[str, Any]]] = {}


class MemoryService:
    """Manages conversational turn persistence and recent session context retrieval."""

    def __init__(self, default_ttl_seconds: int = 86400) -> None:
        self.default_ttl = default_ttl_seconds

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a conversation message to the session's short-term memory list."""
        if not session_id or not content:
            return

        msg_record = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        serialized = json.dumps(msg_record, default=str)
        key = f"zico:session:{session_id}:messages"

        try:
            r = await get_redis()
            pipe = r.pipeline()
            pipe.rpush(key, serialized)
            pipe.ltrim(key, -30, -1)  # Keep last 30 turns max
            pipe.expire(key, self.default_ttl)
            await pipe.execute()
            return
        except Exception as exc:
            logger.debug(f"Redis memory append fallback to local: {exc}")

        # Local fallback
        history = _LOCAL_SESSION_MEMORY.setdefault(session_id, [])
        history.append(msg_record)
        if len(history) > 30:
            _LOCAL_SESSION_MEMORY[session_id] = history[-30:]

    async def get_recent_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieves the recent conversation history up to limit messages."""
        if not session_id:
            return []

        key = f"zico:session:{session_id}:messages"
        try:
            r = await get_redis()
            raw_items = await r.lrange(key, -limit, -1)
            if raw_items:
                return [json.loads(item) for item in raw_items]
        except Exception as exc:
            logger.debug(f"Redis memory retrieval fallback to local: {exc}")

        # Local fallback
        local_history = _LOCAL_SESSION_MEMORY.get(session_id, [])
        return local_history[-limit:]

    async def clear_session(self, session_id: str) -> None:
        """Clears memory for a specific session."""
        if not session_id:
            return

        key = f"zico:session:{session_id}:messages"
        try:
            r = await get_redis()
            await r.delete(key)
        except Exception:
            pass

        _LOCAL_SESSION_MEMORY.pop(session_id, None)


_memory_service_instance: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Returns singleton MemoryService instance."""
    global _memory_service_instance
    if _memory_service_instance is None:
        _memory_service_instance = MemoryService()
    return _memory_service_instance
