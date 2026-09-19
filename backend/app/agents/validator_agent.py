"""
Workflow Validation Agent for ZICO.

This module provides the deterministic validation layer in the ZICO travel operations
architecture. It inspects the application state produced by upstream agents (Router,
Flight, Research) to verify structural integrity, logical consistency, and source
provenance before state is passed to the final response generation layer.

Architecture:
    User -> RouterAgent -> Specialized Agent (Flight/Research) -> ValidatorAgent -> ResponseAgent

Design Principles:
    - Pure Determinism: Enforces strict, deterministic rule-based checks without external
      network, tool, or database dependencies.
    - No Fact Hallucination: Detects contradictions and missing data without inventing
      or guessing travel facts (flights, dates, rules, or URLs).
    - Separation of Concerns: Keeps tool/runtime execution errors (`errors`) strictly separate
      from domain/structural validation failures (`validation_errors`).
    - Idempotent & Deterministic: Running the validator repeatedly on the same state produces
      an identical validation status and consistently ordered validation errors.
    - State Preservation: Updates only validation-owned keys (`validation_status`, `validation_errors`,
      `active_agent`, `completed_agents`), leaving all unrelated state fields intact.
    - LangGraph Ready: Asynchronous and synchronous entry points compatible with workflow nodes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from app.core.logging import get_logger
from app.core.state import TravelState

logger = get_logger(__name__)

# The four authoritative operational intents supported by ZICO
ALLOWED_INTENTS: Set[str] = {"flight", "research", "general_travel", "unsupported"}

# Date formats accepted for basic structural validation
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y")


def _is_valid_url(url: Any) -> bool:
    """
    Validate that a given value is a structurally plausible HTTP or HTTPS URL.
    Does NOT perform external network requests.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _parse_date(date_str: Any) -> Optional[datetime]:
    """
    Attempt to parse a date string against accepted date formats.
    Returns datetime instance if parseable, or None if invalid or malformed.
    """
    if not date_str or not isinstance(date_str, str) or not date_str.strip():
        return None
    cleaned = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Validator Agent Implementation
# ---------------------------------------------------------------------------


class ValidatorAgent:
    """
    Deterministic validation agent verifying structural integrity and logical consistency.
    """

    def _validate_session_and_query(self, state: TravelState, errors: List[str]) -> None:
        """Validate presence and basic structure of user_query."""
        user_query = (
            state.get("user_query")
            if isinstance(state, dict)
            else getattr(state, "user_query", None)
        )
        if user_query is None or not isinstance(user_query, str) or not user_query.strip():
            errors.append("Missing or empty user_query in TravelState.")

    def _validate_intent(self, state: TravelState, errors: List[str]) -> None:
        """Validate that intent is one of the supported operational categories."""
        intent = state.get("intent") if isinstance(state, dict) else getattr(state, "intent", None)
        if intent is None or intent not in ALLOWED_INTENTS:
            errors.append(
                f"Invalid or missing intent '{intent}'. Expected one of: "
                f"{', '.join(sorted(ALLOWED_INTENTS))}."
            )

    def _validate_travel_parameters(self, state: TravelState, errors: List[str]) -> None:
        """Validate dates, passenger counts, and route consistency."""
        # 1. Passenger Count Validation
        passengers = (
            state.get("passengers")
            if isinstance(state, dict)
            else getattr(state, "passengers", None)
        )
        if passengers is not None:
            if not isinstance(passengers, int) or isinstance(passengers, bool) or passengers < 1:
                errors.append(
                    f"Invalid passenger count: passengers must be an integer >= 1, got {passengers}."
                )

        # 2. Date Relationship Validation
        dep_str = (
            state.get("departure_date")
            if isinstance(state, dict)
            else getattr(state, "departure_date", None)
        )
        ret_str = (
            state.get("return_date")
            if isinstance(state, dict)
            else getattr(state, "return_date", None)
        )

        if dep_str and ret_str:
            dep_dt = _parse_date(dep_str)
            ret_dt = _parse_date(ret_str)
            if dep_dt and ret_dt and ret_dt < dep_dt:
                errors.append(
                    f"Invalid date relationship: return_date ({ret_str}) cannot be earlier than "
                    f"departure_date ({dep_str})."
                )

        # 3. Route Contradiction Validation (for flight intent)
        intent = state.get("intent") if isinstance(state, dict) else getattr(state, "intent", None)
        if intent == "flight":
            origin = (
                state.get("origin") if isinstance(state, dict) else getattr(state, "origin", None)
            )
            destination = (
                state.get("destination")
                if isinstance(state, dict)
                else getattr(state, "destination", None)
            )
            if origin and destination and isinstance(origin, str) and isinstance(destination, str):
                orig_clean = origin.strip().upper()
                dest_clean = destination.strip().upper()
                if orig_clean == dest_clean:
                    errors.append(
                        f"Contradictory route: origin and destination cannot be identical ('{origin.strip()}')."
                    )

    def _validate_flight_operations(self, state: TravelState, errors: List[str]) -> None:
        """Validate flight state consistency and result structure for flight requests."""
        flight_status = (
            state.get("flight_status")
            if isinstance(state, dict)
            else getattr(state, "flight_status", None)
        )

        if flight_status == "error":
            errors.append("Flight operation encountered an execution or provider error.")
            return
        elif flight_status in (None, "not_requested"):
            errors.append("Flight operation was not executed or is missing status.")
            return
        elif flight_status == "no_results":
            # Valid state: successful provider query that yielded 0 records
            return
        elif flight_status == "success":
            # Validate flight_results structure
            results = state.get("flight_results", [])
            if not isinstance(results, list):
                errors.append("Flight results must be a list of records.")
                return

            for idx, item in enumerate(results):
                if not isinstance(item, dict):
                    errors.append(
                        f"Malformed flight result at index {idx}: expected dictionary, got {type(item).__name__}."
                    )
                    continue

                # Ensure minimum structural identity exists
                has_id = bool(item.get("flight_iata") or item.get("flight_number"))
                has_route = bool(
                    item.get("departure") or item.get("arrival") or item.get("airline_name")
                )
                if not has_id and not has_route:
                    errors.append(
                        f"Malformed flight result at index {idx}: missing flight identifier or route details."
                    )

    def _validate_research_operations(self, state: TravelState, errors: List[str]) -> None:
        """Validate research state consistency, result structure, and source provenance."""
        research_status = (
            state.get("research_status")
            if isinstance(state, dict)
            else getattr(state, "research_status", None)
        )

        if research_status == "error":
            errors.append("Research operation encountered an execution or provider error.")
            return
        elif research_status in (None, "not_requested"):
            errors.append("Research operation was not executed or is missing status.")
            return
        elif research_status == "no_results":
            # Valid state: successful search that returned 0 matches
            return
        elif research_status == "success":
            # Validate research_results structure and source provenance
            results = state.get("research_results", [])
            if not isinstance(results, list):
                errors.append("Research results must be a list of findings.")
                return

            for idx, item in enumerate(results):
                if not isinstance(item, dict):
                    errors.append(
                        f"Malformed research result at index {idx}: expected dictionary, got {type(item).__name__}."
                    )
                    continue

                # 1. Source URL must be structurally valid
                url = item.get("url")
                if not _is_valid_url(url):
                    errors.append(
                        f"Research result at index {idx} is missing a valid HTTP/HTTPS source URL (got {url!r})."
                    )

                # 2. Content or summary must be present
                content = item.get("content")
                if content is None or not isinstance(content, str):
                    errors.append(f"Research result at index {idx} is missing text content.")

    def validate(self, state: TravelState) -> Dict[str, Any]:
        """
        Execute deterministic validation pipeline across TravelState.

        Args:
            state: Active TravelState dictionary.

        Returns:
            Partial state update dictionary containing `validation_status`,
            `validation_errors`, `active_agent`, and `completed_agents`.
        """
        logger.info("Starting workflow state validation")
        errors: List[str] = []

        # 1. Request / Session Integrity
        self._validate_session_and_query(state, errors)

        # 2. Intent Validity
        self._validate_intent(state, errors)

        # 3. Travel Parameters & Contradictions
        self._validate_travel_parameters(state, errors)

        # 4. Operation-Specific State Consistency
        intent = state.get("intent") if isinstance(state, dict) else getattr(state, "intent", None)
        if intent == "flight":
            self._validate_flight_operations(state, errors)
        elif intent == "research":
            self._validate_research_operations(state, errors)
        # Note: 'general_travel' and 'unsupported' require no specialized tool outputs

        # 5. Deterministic Deduplication & Ordering
        deduped_errors: List[str] = list(dict.fromkeys(errors))
        status = "passed" if len(deduped_errors) == 0 else "failed"

        completed_agents = list(state.get("completed_agents", []))
        if "validator_agent" not in completed_agents:
            completed_agents.append("validator_agent")

        if status == "passed":
            logger.info("Validation passed successfully with 0 errors")
        else:
            logger.warning(
                "Validation failed with %d error(s): %s", len(deduped_errors), deduped_errors
            )

        return {
            "validation_status": status,
            "validation_errors": deduped_errors,
            "active_agent": "validator_agent",
            "completed_agents": completed_agents,
        }

    async def avalidate(self, state: TravelState) -> Dict[str, Any]:
        """
        Asynchronously execute deterministic validation on TravelState.

        Args:
            state: Active TravelState dictionary.

        Returns:
            Partial state update dictionary.
        """
        return self.validate(state)


# ---------------------------------------------------------------------------
# Module-level Convenience Entry Points
# ---------------------------------------------------------------------------


async def run_validator_agent(state: TravelState) -> Dict[str, Any]:
    """
    Asynchronous entry point for ZICO Validator Agent (LangGraph node compatible).

    Args:
        state: Active TravelState dictionary.

    Returns:
        Partial state update dictionary containing updated `validation_status`
        and `validation_errors`.
    """
    agent = ValidatorAgent()
    return await agent.avalidate(state)


def validate_state(state: TravelState) -> Dict[str, Any]:
    """
    Synchronous entry point for ZICO Validator Agent.

    Args:
        state: Active TravelState dictionary.

    Returns:
        Partial state update dictionary.
    """
    agent = ValidatorAgent()
    return agent.validate(state)
