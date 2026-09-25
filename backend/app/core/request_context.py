"""
Request-local execution context management for ZICO.

Provides isolated, thread-safe, and asynchronous task-local storage for
`request_id` and `session_id` tracing via Python's standard-library `contextvars`.

Design Principles:
    - Pure Standard Library: Uses Python's native `contextvars` module.
    - Zero Cross-Request Leakage: Context is strictly bound to the active async task/thread.
    - Privacy & Security: Stores only non-sensitive operational identifiers.
      Never stores prompts, full messages, credentials, API keys, or provider tokens.
    - Deterministic Cleanup: Intended to be paired with middleware finally-blocks.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Dict, Optional, Tuple

# Context variable identifiers for active execution context
_request_id_var: ContextVar[Optional[str]] = ContextVar("zico_request_id", default=None)
_session_id_var: ContextVar[Optional[str]] = ContextVar("zico_session_id", default=None)

__all__ = [
    "set_request_context",
    "get_request_context",
    "clear_request_context",
    "get_request_id",
    "get_session_id",
    "set_request_id",
    "set_session_id",
]


def set_request_context(
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[Token[Optional[str]], Token[Optional[str]]]:
    """
    Establish active request and session identifiers for the current execution context.

    Args:
        request_id: Unique operational request trace identifier (e.g. 'req_123').
        session_id: Traveler conversation session identifier (e.g. 'sess_456').

    Returns:
        A tuple of (request_id_token, session_id_token) suitable for resetting.
    """
    token_req = _request_id_var.set(request_id)
    token_sess = _session_id_var.set(session_id)
    return token_req, token_sess


def set_request_id(request_id: Optional[str]) -> Token[Optional[str]]:
    """
    Set the operational request trace identifier for the current context.

    Args:
        request_id: Unique request trace identifier.

    Returns:
        ContextVar token for the modified request_id.
    """
    return _request_id_var.set(request_id)


def set_session_id(session_id: Optional[str]) -> Token[Optional[str]]:
    """
    Set the traveler conversation session identifier for the current context.

    Args:
        session_id: User/conversation session identifier.

    Returns:
        ContextVar token for the modified session_id.
    """
    return _session_id_var.set(session_id)


def get_request_id() -> Optional[str]:
    """
    Retrieve the active request trace identifier.

    Returns:
        The current request ID string, or None if no context has been established.
    """
    return _request_id_var.get()


def get_session_id() -> Optional[str]:
    """
    Retrieve the active traveler conversation session identifier.

    Returns:
        The current session ID string, or None if no context has been established.
    """
    return _session_id_var.get()


def get_request_context() -> Dict[str, Optional[str]]:
    """
    Obtain a snapshot dictionary of the current request execution context.

    Returns:
        Dictionary containing 'request_id' and 'session_id'.
    """
    return {
        "request_id": get_request_id(),
        "session_id": get_session_id(),
    }


def clear_request_context() -> None:
    """
    Clear all active context variables for the current execution task.

    Guarantees that subsequent operations on the same worker thread/task do not
    inherit stale request identifiers.
    """
    _request_id_var.set(None)
    _session_id_var.set(None)
