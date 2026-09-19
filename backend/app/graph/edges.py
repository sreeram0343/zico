"""
LangGraph Conditional Routing Layer for ZICO Travel Operations.

This module provides the deterministic routing decision logic executed immediately
after the Intent Router Agent in the ZICO LangGraph workflow.

Routing Logic:
    - 'flight'         -> 'flight'   (Delegates to Flight Operations Agent)
    - 'research'       -> 'research' (Delegates to Travel Research Agent)
    - 'general_travel' -> 'research' (Broad inquiries leverage Research Agent for MVP)
    - 'unsupported'    -> 'response' (Direct terminal path explaining out-of-scope request)
    - invalid/missing  -> 'response' (Safe terminal fallback communicating limitations)

Architectural Principles:
    - Pure Decision Logic: Evaluates `state["intent"]` and returns the target node name.
    - No Execution: Does not invoke nodes, tools, LLMs, or external APIs.
    - Decoupled from Workflow: Does not construct, register, or compile the graph workflow.
    - Immutability: Operates on `TravelState` without mutating state fields.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional

from app.core.logging import get_logger
from app.core.state import TravelState

logger = get_logger(__name__)

# Authoritative intent-to-node mapping for ZICO's routing layer
_ROUTE_MAP = {
    "flight": "flight",
    "research": "research",
    "general_travel": "research",
    "unsupported": "response",
}

# Immutable public routing map to prevent runtime mutation
ROUTE_MAP: Mapping[str, str] = MappingProxyType(_ROUTE_MAP)

# Safe terminal node fallback for unhandled, missing, or malformed intents
DEFAULT_FALLBACK_NODE: str = "response"

__all__ = [
    "route_after_router",
    "ROUTE_MAP",
    "DEFAULT_FALLBACK_NODE",
]


def route_after_router(state: TravelState) -> str:
    """
    Determine the next LangGraph node identifier following the router node.

    Inspects the `intent` field in `TravelState`, performs defensive normalization,
    and returns the target node identifier. Unknown, missing, or empty intents
    safely route to the 'response' node.

    Args:
        state: Active graph TravelState dictionary.

    Returns:
        Target node identifier: 'flight', 'research', or 'response'.
    """
    raw_intent: Optional[str] = state.get("intent") if isinstance(state, dict) else None

    if not raw_intent or not isinstance(raw_intent, str):
        logger.info(
            "Missing or non-string intent encountered; routing to '%s'", DEFAULT_FALLBACK_NODE
        )
        return DEFAULT_FALLBACK_NODE

    normalized_intent = raw_intent.strip().lower()

    target_node = ROUTE_MAP.get(normalized_intent, DEFAULT_FALLBACK_NODE)
    logger.info(
        "Routing intent '%s' (normalized: '%s') to node '%s'",
        raw_intent,
        normalized_intent,
        target_node,
    )

    return target_node
