"""
Flight Operations Agent for ZICO.

This module provides the specialized Flight Agent responsible for:
    - Inspecting flight-related travel parameters from `TravelState`.
    - Extracting flight designators (e.g. "EK522", "AI123") or origin/destination pairs.
    - Resolving natural-language city and airport queries via `LocationResolver`.
    - Executing flight status and search lookups via `AviationStackClient`.
    - Storing normalized flight records and status updates into `TravelState`.

Architecture:
    TravelState -> FlightAgent -> LocationResolver -> AviationStackClient -> Updated TravelState

Design Principles:
    - State Compliance: Strictly updates fields owned by flight operations (`flight_query`,
      `flight_results`, `flight_status`, `active_agent`, `completed_agents`, `errors`).
    - Intent Guard: Rejects non-flight requests (`intent != 'flight'`) without executing tools.
    - Robust Resolution: Handles flight numbers, airport codes, cities, and dates cleanly.
    - Ambiguity & Error Resilient: Distinctly tracks 'success', 'no_results', and 'error'.
    - LangGraph Ready: Asynchronous entry point for workflow graphs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import get_logger
from app.core.state import TravelState
from app.tools.aviationstack import (
    AviationStackClient,
    AviationStackError,
    NormalizedFlight,
)
from app.tools.location import LocationResolver, resolve_location

logger = get_logger(__name__)

# Common English prepositions/words to exclude from standalone airline designator matching
STOPWORDS = {
    "ON",
    "IN",
    "TO",
    "AT",
    "BY",
    "NO",
    "IS",
    "IT",
    "DO",
    "GO",
    "ME",
    "MY",
    "HE",
    "WE",
    "US",
    "OR",
    "IF",
    "SO",
    "AS",
}

# Explicit flight prefix pattern (e.g. "flight EK522", "flt 522", "flight no 123")
EXPLICIT_FLIGHT_REGEX = re.compile(
    r"\b(?:flight|flt)\s+(?:number\s+|no\.?\s+|#\s*)?([A-Za-z]{2,3}\s*\d{1,4}|\d{1,4})\b",
    re.IGNORECASE,
)

# Standalone airline designator + flight number (e.g. EK522, AI123, BA 117)
STANDALONE_FLIGHT_REGEX = re.compile(r"\b([A-Za-z]{2,3})\s*(\d{1,4}[A-Za-z]?)\b")

# Regex for extracting "from <origin> to <destination>"
ROUTE_REGEX = re.compile(
    r"\bfrom\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+?)(?:\s+on|\s+dated|\s+for|[\s\.]*$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Flight Agent Implementation
# ---------------------------------------------------------------------------


class FlightAgent:
    """
    Specialized agent for processing flight status, tracking, and search operations.
    """

    def __init__(
        self,
        aviation_client: Optional[AviationStackClient] = None,
        location_resolver: Optional[LocationResolver] = None,
    ) -> None:
        """
        Initialize the Flight Agent.

        Args:
            aviation_client: Optional pre-configured AviationStackClient (useful for testing).
            location_resolver: Optional pre-configured LocationResolver (useful for testing).
        """
        self.aviation_client = aviation_client
        self.location_resolver = location_resolver

    def _resolve_location_info(
        self, location_query: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve a city or airport query into (iata_code, status).

        Returns:
            Tuple of (iata_code or None, resolution_status or None).
        """
        if not location_query or not isinstance(location_query, str) or not location_query.strip():
            return None, None

        cleaned = location_query.strip()
        # Avoid duplicate resolution if already a standard 3-letter IATA code
        if len(cleaned) == 3 and cleaned.isalpha():
            return cleaned.upper(), "resolved"

        if self.location_resolver:
            res = self.location_resolver.resolve(cleaned)
        else:
            res = resolve_location(cleaned)

        if res.status == "resolved" and res.iata_code:
            logger.info("Resolved location %r -> %s", cleaned, res.iata_code)
            return res.iata_code, "resolved"

        logger.warning("Location resolution for %r produced status=%s", cleaned, res.status)
        return None, res.status

    def _extract_flight_params(self, state: TravelState) -> Dict[str, Any]:
        """
        Extract flight search parameters from TravelState and user query.
        """
        params: Dict[str, Any] = {}

        # 1. Check existing flight_query in state
        existing_query = state.get("flight_query") or {}
        if isinstance(existing_query, dict):
            params.update(existing_query)

        user_query = str(state.get("user_query") or "").strip().rstrip(".?! ")
        origin = state.get("origin")
        destination = state.get("destination")
        departure_date = state.get("departure_date")

        # 2. Extract flight number / IATA code if not already in params
        if not params.get("flight_iata") and not params.get("flight_number"):
            # Option A: Check for explicit "flight <code/number>" phrasing
            explicit_match = EXPLICIT_FLIGHT_REGEX.search(user_query)
            if explicit_match:
                token = explicit_match.group(1).replace(" ", "").upper()
                if token.isdigit():
                    params["flight_number"] = token
                else:
                    params["flight_iata"] = token
                    num_match = re.search(r"\d+", token)
                    if num_match:
                        params["flight_number"] = num_match.group(0)
                logger.info(
                    "Extracted flight identifier from explicit phrasing: %s",
                    params.get("flight_iata") or params.get("flight_number"),
                )
            else:
                # Option B: Check for standalone airline code + number (excluding stopwords and dates/years)
                standalone_match = STANDALONE_FLIGHT_REGEX.search(user_query)
                if standalone_match:
                    code = standalone_match.group(1).upper()
                    num = standalone_match.group(2)
                    is_year = len(num) == 4 and num.startswith(("19", "20"))
                    if code not in STOPWORDS and not is_year:
                        params["flight_iata"] = f"{code}{num}"
                        params["flight_number"] = num
                        logger.info(
                            "Extracted standalone flight identifier: %s", params["flight_iata"]
                        )

        # 3. Extract route from state or query
        if not origin or not destination:
            route_match = ROUTE_REGEX.search(user_query)
            if route_match:
                if not origin:
                    origin = route_match.group(1).strip()
                if not destination:
                    destination = route_match.group(2).strip()

        # 4. Resolve origin / destination IATA codes
        if origin and not params.get("dep_iata"):
            dep_iata, origin_status = self._resolve_location_info(origin)
            if dep_iata:
                params["dep_iata"] = dep_iata
            else:
                params["raw_origin"] = origin
                params["origin_status"] = origin_status

        if destination and not params.get("arr_iata"):
            arr_iata, dest_status = self._resolve_location_info(destination)
            if arr_iata:
                params["arr_iata"] = arr_iata
            else:
                params["raw_destination"] = destination
                params["destination_status"] = dest_status

        if departure_date and not params.get("flight_date"):
            params["flight_date"] = departure_date

        return params

    async def execute(self, state: TravelState) -> Dict[str, Any]:
        """
        Execute the flight operations agent on the provided state.

        Args:
            state: Active TravelState.

        Returns:
            Partial state update dictionary containing updated `flight_query`,
            `flight_results`, `flight_status`, `active_agent`, and `completed_agents`.
        """
        current_errors = list(state.get("errors", []))
        completed_agents = list(state.get("completed_agents", []))
        if "flight_agent" not in completed_agents:
            completed_agents.append("flight_agent")

        # 1. Intent Validation Guard
        intent = state.get("intent")
        if intent != "flight":
            logger.warning("Flight Agent invoked with non-flight intent %r", intent)
            return {
                "flight_status": "error",
                "active_agent": "flight_agent",
                "completed_agents": completed_agents,
                "errors": current_errors
                + [f"Flight Agent invoked with invalid intent '{intent}'. Expected 'flight'."],
            }

        logger.info("Flight Agent executing flight operations workflow")

        # 2. Extract and Normalize Flight Parameters
        flight_params = self._extract_flight_params(state)

        # Check if an explicitly requested origin or destination failed resolution
        raw_origin = flight_params.get("raw_origin")
        raw_dest = flight_params.get("raw_destination")

        if raw_origin or raw_dest:
            err_details = []
            if raw_origin:
                st = flight_params.get("origin_status", "unresolved")
                if st == "ambiguous":
                    err_details.append(
                        f"origin '{raw_origin}' is ambiguous and matches multiple airports"
                    )
                elif st == "not_found":
                    err_details.append(f"origin '{raw_origin}' could not be resolved (not_found)")
                else:
                    err_details.append(f"could not resolve origin '{raw_origin}'")

            if raw_dest:
                st = flight_params.get("destination_status", "unresolved")
                if st == "ambiguous":
                    err_details.append(
                        f"destination '{raw_dest}' is ambiguous and matches multiple airports"
                    )
                elif st == "not_found":
                    err_details.append(
                        f"destination '{raw_dest}' could not be resolved (not_found)"
                    )
                else:
                    err_details.append(f"could not resolve destination '{raw_dest}'")

            msg = f"Flight route resolution failed: {', '.join(err_details)}."
            logger.error("Flight Agent failed: %s", msg)
            return {
                "flight_query": flight_params,
                "flight_results": [],
                "flight_status": "error",
                "active_agent": "flight_agent",
                "completed_agents": completed_agents,
                "errors": current_errors + [msg],
            }

        # Check if we have minimum viable search parameters
        has_flight_id = bool(flight_params.get("flight_iata") or flight_params.get("flight_number"))
        has_route = bool(flight_params.get("dep_iata") and flight_params.get("arr_iata"))

        if not has_flight_id and not has_route:
            msg = "Insufficient flight information to query AviationStack. Please provide a flight number (e.g. 'EK522') or valid route (e.g. 'from TRV to DXB')."
            logger.error("Flight Agent failed: %s", msg)
            return {
                "flight_query": flight_params,
                "flight_results": [],
                "flight_status": "error",
                "active_agent": "flight_agent",
                "completed_agents": completed_agents,
                "errors": current_errors + [msg],
            }

        # 3. Query AviationStack
        client = self.aviation_client
        manage_client = False
        if client is None:
            client = AviationStackClient()
            manage_client = True

        try:
            logger.info(
                "Querying AviationStack with parameters: %s",
                {k: v for k, v in flight_params.items() if not k.startswith("raw_")},
            )
            response = await client.get_flights(
                flight_iata=flight_params.get("flight_iata"),
                flight_number=flight_params.get("flight_number"),
                dep_iata=flight_params.get("dep_iata"),
                arr_iata=flight_params.get("arr_iata"),
                flight_date=flight_params.get("flight_date"),
            )
        except AviationStackError as exc:
            logger.error("AviationStack provider error: %s", exc)
            return {
                "flight_query": flight_params,
                "flight_results": [],
                "flight_status": "error",
                "active_agent": "flight_agent",
                "completed_agents": completed_agents,
                "errors": current_errors + [f"AviationStack provider error: {exc}"],
            }
        except Exception as exc:
            logger.error("Unexpected error during flight lookup: %s", exc)
            return {
                "flight_query": flight_params,
                "flight_results": [],
                "flight_status": "error",
                "active_agent": "flight_agent",
                "completed_agents": completed_agents,
                "errors": current_errors + [f"Unexpected flight lookup error: {exc}"],
            }
        finally:
            if manage_client:
                await client.close()

        # 4. Normalize and Store Flight Results
        normalized_data: List[Dict[str, Any]] = [
            f.model_dump() if isinstance(f, NormalizedFlight) else dict(f) for f in response.data
        ]

        status = "success" if normalized_data else "no_results"
        logger.info(
            "Flight Agent completed lookup (results=%d, status=%s)", len(normalized_data), status
        )

        return {
            "flight_query": flight_params,
            "flight_results": normalized_data,
            "flight_status": status,
            "active_agent": "flight_agent",
            "completed_agents": completed_agents,
        }


# ---------------------------------------------------------------------------
# Module-level Convenience Entry Points
# ---------------------------------------------------------------------------


async def run_flight_agent(
    state: TravelState,
    aviation_client: Optional[AviationStackClient] = None,
    location_resolver: Optional[LocationResolver] = None,
) -> Dict[str, Any]:
    """
    Asynchronous entry point for ZICO Flight Agent (LangGraph node compatible).

    Args:
        state: Active TravelState dictionary.
        aviation_client: Optional client override for testing or custom connection.
        location_resolver: Optional location resolver override.

    Returns:
        Partial state update dictionary.
    """
    agent = FlightAgent(
        aviation_client=aviation_client,
        location_resolver=location_resolver,
    )
    return await agent.execute(state)
