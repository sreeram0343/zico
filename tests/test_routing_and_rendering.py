"""
Regression tests for ZICO Routing, State Isolation, Entity Hallucination Guards,
Booking Capabilities, and Message Rendering.

Covers:
1. Hotel request: "Hotels in Pune under 10k" -> HOTEL_SEARCH (not flight)
2. Flight request: "Find flights from Pune to Dubai tomorrow" -> FLIGHT_SEARCH
3. Destination research: "What are the best places to visit in Pune?" -> DESTINATION_RESEARCH
4. Multi-turn State Isolation: "Flights from Pune to Dubai" followed by "Hotels in Pune under 10k"
   verifying no Dubai or flight contamination leaks into request 2.
5. Invalid routing rejection / controlled handling.
6. Missing flight destination -> clarification question, no hallucinated destination.
7. Missing flight origin -> clarification question, no hallucinated origin.
8. Safe Markdown formatting (no literal asterisks).
9. Absence of internal graph debug node names in responses.
10. Inappropriate booking language suppression (safe capability language only).
11. No fabricated travel entities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure backend directory is in sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.agents.response_agent import generate_deterministic_fallback  # noqa: E402
from app.agents.router_agent import classify_intent_heuristically  # noqa: E402
from app.core.state import create_initial_state  # noqa: E402
from app.graph.edges import route_after_router, validate_route  # noqa: E402
from app.graph.engine import _extract_airports, flight_search_worker_node  # noqa: E402
from app.graph.supervisor import _classify_intent_heuristically as supervisor_classify  # noqa: E402


class TestIntentClassification:
    """Tests 1, 2, and 3: Intent Classification correctness."""

    def test_hotel_request_intent(self):
        """Test 1: 'Hotels in Pune under 10k' must classify as HOTEL_SEARCH with Pune and 10000 INR."""
        query = "Hotels in Pune under 10k"
        decision = classify_intent_heuristically(query)

        assert decision.intent == "HOTEL_SEARCH"
        assert decision.intent != "FLIGHT_SEARCH"
        assert decision.location == "Pune"
        assert decision.budget == "10000 INR"

    def test_flight_request_intent(self):
        """Test 2: 'Find flights from Pune to Dubai tomorrow' must classify as FLIGHT_SEARCH."""
        query = "Find flights from Pune to Dubai tomorrow"
        decision = classify_intent_heuristically(query)

        assert decision.intent == "FLIGHT_SEARCH"
        assert decision.origin == "Pune"
        assert decision.destination == "Dubai"

    def test_destination_research_intent(self):
        """Test 3: 'What are the best places to visit in Pune?' must classify as DESTINATION_RESEARCH."""
        query = "What are the best places to visit in Pune?"
        decision = classify_intent_heuristically(query)

        assert decision.intent == "DESTINATION_RESEARCH"
        assert decision.location == "Pune"

    def test_supervisor_heuristics_hotel_routing(self):
        """Supervisor must route hotel inquiries to research_worker, NEVER flight_search_worker."""
        route = supervisor_classify("Hotels in Pune under 10k")
        assert route == "research_worker"
        assert route != "flight_search_worker"

    def test_supervisor_heuristics_flight_routing(self):
        """Supervisor must route flight inquiries to flight_search_worker."""
        route = supervisor_classify("Find flights from Pune to Dubai tomorrow")
        assert route == "flight_search_worker"


class TestStateIsolation:
    """Test 4: State contamination prevention between multi-turn requests."""

    def test_clean_state_isolation(self):
        """
        Request 1: 'Find flights from Pune to Dubai.'
        Request 2: 'Hotels in Pune under 10k.'
        Verify that request 2 contains no Dubai or flight-search information.
        """
        # Request 1: Flight query
        state1 = create_initial_state(
            user_query="Find flights from Pune to Dubai",
            session_id="session_iso_01",
            request_id="req_01",
        )
        decision1 = classify_intent_heuristically(state1["user_query"])
        state1["intent"] = (
            decision1.intent.value if hasattr(decision1.intent, "value") else decision1.intent
        )
        state1["origin"] = decision1.origin
        state1["destination"] = decision1.destination

        assert state1["intent"] == "FLIGHT_SEARCH"
        assert state1["destination"] == "Dubai"

        # Request 2: New Hotel query in same session
        state2 = create_initial_state(
            user_query="Hotels in Pune under 10k",
            session_id="session_iso_01",
            request_id="req_02",
        )
        decision2 = classify_intent_heuristically(state2["user_query"])
        state2["intent"] = (
            decision2.intent.value if hasattr(decision2.intent, "value") else decision2.intent
        )
        state2["location"] = decision2.location
        state2["budget"] = decision2.budget

        # Verify state2 is completely clean of state1 fields
        assert state2["intent"] == "HOTEL_SEARCH"
        assert state2["location"] == "Pune"
        assert state2["budget"] == "10000 INR"
        assert state2.get("destination") is None
        assert state2.get("origin") is None
        assert "dubai" not in str(state2).lower()

        # Check response fallback generation for request 2
        response_text = generate_deterministic_fallback(state2)
        assert "dubai" not in response_text.lower()
        assert "flight" not in response_text.lower()
        assert "pune" in response_text.lower()


class TestRouteValidation:
    """Test 5: Controlled routing and validation."""

    def test_hotel_search_routes_to_research(self):
        """HOTEL_SEARCH must route to research, never flight."""
        state = create_initial_state(
            user_query="Hotels in Pune under 10k",
            session_id="test_sess",
            request_id="test_req",
        )
        state["intent"] = "HOTEL_SEARCH"
        route = route_after_router(state)
        assert route == "research"
        assert route != "flight"

    def test_flight_search_routes_to_flight(self):
        """FLIGHT_SEARCH must route to flight."""
        state = create_initial_state(
            user_query="Flights from Pune to Dubai",
            session_id="test_sess",
            request_id="test_req",
        )
        state["intent"] = "FLIGHT_SEARCH"
        route = route_after_router(state)
        assert route == "flight"

    def test_invalid_routing_controlled(self):
        """Invalid route must return controlled error and never fall back to flight."""
        state = create_initial_state(
            user_query="Random gibberish inquiry",
            session_id="test_sess",
            request_id="test_req",
        )
        state["intent"] = "UNSUPPORTED"
        route = route_after_router(state)
        assert route == "response"
        assert route != "flight"

    def test_validate_route_helper(self):
        """validate_route must validate allowed destinations and raise or reject invalid."""
        assert validate_route("HOTEL_SEARCH", "research") is True
        assert validate_route("FLIGHT_SEARCH", "flight") is True
        assert validate_route("HOTEL_SEARCH", "flight") is False


class TestEntityHallucinationGuards:
    """Tests 6, 7, and 11: No fabricated travel entities."""

    def test_missing_flight_destination_clarification(self):
        """Test 6: Missing flight destination must trigger clarification without inventing destination."""
        origin, dest, origin_name, dest_name = _extract_airports("Find me a flight from Pune")
        assert origin == "PNQ"
        assert dest == ""  # Never default to DXB or BOM

        state = {
            "messages": [{"role": "user", "content": "Find me a flight from Pune"}],
            "itinerary": [],
        }
        result = flight_search_worker_node(state)
        msg_content = result["messages"][0].content

        assert "destination" in msg_content.lower()
        assert "dubai" not in msg_content.lower()

    def test_missing_flight_origin_clarification(self):
        """Test 7: Missing flight origin must trigger clarification without inventing origin (BOM)."""
        origin, dest, origin_name, dest_name = _extract_airports("Find me a flight to Dubai")
        assert dest == "DXB"
        assert origin == ""  # Never default to BOM

        state = {
            "messages": [{"role": "user", "content": "Find me a flight to Dubai"}],
            "itinerary": [],
        }
        result = flight_search_worker_node(state)
        msg_content = result["messages"][0].content

        assert "origin" in msg_content.lower() or "depart" in msg_content.lower()
        assert "mumbai" not in msg_content.lower()

    def test_no_fabricated_fallback_flights(self):
        """Test 11: Flight search with no live results must not invent fake airline names."""
        state = {
            "messages": [
                {
                    "role": "user",
                    "content": "Find flights from Pune to Dubai on 2026-10-15",
                }
            ],
            "itinerary": [],
        }
        with patch("app.graph.engine.search_flights") as mock_search:
            mock_search.invoke.return_value = []
            result = flight_search_worker_node(state)
            msg_content = result["messages"][0].content

            # Must state no live options found, not invent Air India AI-367
            assert "no live flight options were found" in msg_content.lower()
            assert "spicejet" not in msg_content.lower()
            assert "indigo" not in msg_content.lower()


class TestResponseSerializationAndNodeHiding:
    """Tests 8, 9, 10: Formatting, node name hiding, and booking claim prevention."""

    def test_debug_node_names_not_in_response(self):
        """Test 9: Internal LangGraph node labels must NEVER appear in final responses."""
        state = create_initial_state(
            user_query="Hotels in Pune under 10k",
            session_id="test_sess",
            request_id="test_req",
        )
        state["intent"] = "HOTEL_SEARCH"
        state["location"] = "Pune"
        state["budget"] = "10000 INR"

        response = generate_deterministic_fallback(state)

        for forbidden in [
            "INPUT_NODE",
            "SUPERVISOR_NODE",
            "FLIGHT_NODE",
            "RESEARCH_NODE",
            "VALIDATOR_NODE",
            "input_node",
            "supervisor_node",
            "flight_search_worker",
        ]:
            assert forbidden not in response

    def test_no_inappropriate_booking_claims(self):
        """Test 10: ZICO must never claim it can reserve/book a flight without booking tool."""
        state = create_initial_state(
            user_query="Find flights from Pune to Dubai",
            session_id="test_sess",
            request_id="test_req",
        )
        state["intent"] = "FLIGHT_SEARCH"
        state["flight_status"] = "success"
        state["flight_results"] = [
            {
                "flight_iata": "AI101",
                "airline_name": "Air India",
                "flight_status": "scheduled",
                "departure": {"iata": "PNQ", "scheduled": "2026-10-15T10:00:00"},
                "arrival": {"iata": "DXB", "scheduled": "2026-10-15T13:00:00"},
            }
        ]

        response = generate_deterministic_fallback(state)

        # Must not claim booking capabilities
        assert "i have reserved" not in response.lower()
        assert "booking confirmed" not in response.lower()
        # Must offer safe alternatives
        assert "compare" in response.lower() or "add this to your itinerary" in response.lower()

    def test_hotel_response_clean_formatting(self):
        """Test 8 & 1: Hotel response formats cleanly with markdown bolding."""
        state = create_initial_state(
            user_query="Hotels in Pune under 10k",
            session_id="test_sess",
            request_id="test_req",
        )
        state["intent"] = "HOTEL_SEARCH"
        state["location"] = "Pune"
        state["budget"] = "10000 INR"
        state["research_status"] = "success"
        state["research_results"] = [
            {
                "title": "Hotel Central Pune",
                "url": "https://example.com/hotel",
                "content": "Comfortable rooms near city center under 5000 INR per night.",
            }
        ]

        response = generate_deterministic_fallback(state)
        assert "**Hotel Central Pune**" in response
        assert "**Pune**" in response
        assert "within 10000 INR" in response
