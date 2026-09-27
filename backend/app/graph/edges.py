"""
LangGraph Conditional Routing Layer for ZICO Travel Operations.

This module provides the deterministic routing decision logic executed immediately
after the Intent Router Agent in the ZICO LangGraph workflow.

Routing Logic:
    - 'FLIGHT_SEARCH', 'FLIGHT_STATUS', 'flight' -> 'flight'   (Delegates to Flight Operations Agent)
    - 'HOTEL_SEARCH', 'DESTINATION_RESEARCH', 'ITINERARY_PLANNING',
      'TRAVEL_POLICY', 'LOCATION_QUERY', 'GENERAL_TRAVEL', 'research' -> 'research' (Delegates to Travel Research Agent)
    - 'UNSUPPORTED', 'unsupported' -> 'response' (Direct terminal path explaining out-of-scope request)
    - invalid/missing              -> 'response' (Controlled terminal fallback without silent flight delegation)

Architectural Principles:
    - Pure Decision Logic: Evaluates `state["intent"]` and returns the target node name.
    - Route Validation: Never silently falls back to flight agent upon invalid/unsupported intents.
    - Safe Error Tracking: Logs controlled routing anomalies.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional

from app.core.logging import get_logger
from app.core.state import TravelState

logger = get_logger(__name__)

# Authoritative intent-to-node mapping for ZICO's routing layer
_ROUTE_MAP = {
    # Canonical uppercase intents
    "FLIGHT_SEARCH": "flight",
    "FLIGHT_STATUS": "flight",
    "HOTEL_SEARCH": "research",
    "DESTINATION_RESEARCH": "research",
    "ITINERARY_PLANNING": "research",
    "TRAVEL_POLICY": "research",
    "LOCATION_QUERY": "research",
    "GENERAL_TRAVEL": "research",
    "UNSUPPORTED": "response",
    # Normalized lowercase variants
    "flight_search": "flight",
    "flight_status": "flight",
    "hotel_search": "research",
    "destination_research": "research",
    "itinerary_planning": "research",
    "travel_policy": "research",
    "location_query": "research",
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
    "validate_route",
    "ROUTE_MAP",
    "DEFAULT_FALLBACK_NODE",
]


def route_after_router(state: TravelState) -> str:
    """
    Determine the next LangGraph node identifier following the router node.

    Inspects the `intent` field in `TravelState`, performs defensive normalization,
    and returns the target node identifier. Unknown, missing, or empty intents
    safely route to the 'response' node. Never silently delegates to the flight agent.

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

    cleaned_intent = raw_intent.strip()
    target_node = ROUTE_MAP.get(cleaned_intent) or ROUTE_MAP.get(
        cleaned_intent.lower(), DEFAULT_FALLBACK_NODE
    )

    if target_node == DEFAULT_FALLBACK_NODE and cleaned_intent.lower() not in ("unsupported",):
        logger.warning("Invalid or unmapped intent %r; routing safely to response.", raw_intent)
        if isinstance(state, dict) and "errors" in state:
            state["errors"].append(f"Invalid route for intent '{raw_intent}'.")

    logger.info(
        "Routing intent '%s' (target: '%s')",
        raw_intent,
        target_node,
    )

    return target_node


def validate_route(intent: str, destination: str) -> bool:
    """
    Validate whether an intent is allowed to route to a given destination node.

    Args:
        intent: The classified intent string.
        destination: Target node candidate ('flight', 'research', 'response').

    Returns:
        True if routing is valid, False otherwise.
    """
    if not intent or not isinstance(intent, str):
        return destination == DEFAULT_FALLBACK_NODE

    cleaned = intent.strip()
    expected = (
        ROUTE_MAP.get(cleaned) or ROUTE_MAP.get(cleaned.upper()) or ROUTE_MAP.get(cleaned.lower())
    )
    return expected == destination
