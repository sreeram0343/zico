"""
LangGraph Node Adapter Layer for ZICO Travel Operations.

This module provides thin, asynchronous LangGraph-compatible node functions
wrapping ZICO's core domain agents:
    - router_node: Adapts the Intent Router Agent.
    - flight_node: Adapts the Flight Operations Agent.
    - research_node: Adapts the Travel Research Agent.
    - validator_node: Adapts the Business Rule & Safety Validator Agent.
    - response_node: Adapts the Final Response Generation Agent.

Architectural Principles:
    - Pure Adapters: Nodes contain minimal adapter logic only and delegate all
      business, validation, and LLM operations directly to the underlying agents.
    - State is Source of Truth: Operates strictly on `TravelState` (from `app.core.state`).
    - Tool & Provider Isolation: Nodes never directly execute tools (AviationStack, Tavily,
      LocationResolver) or construct external API clients.
    - Clean Exception Propagation: Meaningful agent exceptions are not swallowed.
    - Graph Separation: Workflow assembly, conditional branching, and graph
      compilation are strictly decoupled and handled in dedicated graph modules.
"""

from __future__ import annotations

from typing import Any, Dict

from app.agents.flight_agent import run_flight_agent
from app.agents.research_agent import run_research_agent
from app.agents.response_agent import run_response_agent
from app.agents.router_agent import aroute_request
from app.agents.validator_agent import run_validator_agent
from app.core.logging import get_logger
from app.core.state import TravelState

logger = get_logger(__name__)

__all__ = [
    "router_node",
    "flight_node",
    "research_node",
    "validator_node",
    "response_node",
]


async def router_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Intent Router Agent.

    Inspects the active `TravelState` and classifies the traveler's request
    into one of the authoritative operational intents ('flight', 'research',
    'general_travel', 'unsupported').

    Args:
        state: Current graph TravelState.

    Returns:
        Partial state update containing the classified `intent` and `active_agent`.
    """
    logger.info("Executing router node")
    update = await aroute_request(state)
    logger.info("Router node completed (intent=%s)", update.get("intent"))
    return update


async def flight_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Flight Operations Agent.

    Extracts flight parameters, resolves airport/city locations, and queries
    AviationStack flight data, writing normalized flight results to state.

    Args:
        state: Current graph TravelState.

    Returns:
        Partial state update containing `flight_query`, `flight_results`,
        `flight_status`, and updated agent tracking.
    """
    logger.info("Executing flight node")
    update = await run_flight_agent(state)
    logger.info("Flight node completed (flight_status=%s)", update.get("flight_status"))
    return update


async def research_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Travel Research Agent.

    Generates optimized search queries and queries Tavily search API,
    storing deduplicated, normalized research findings into state.

    Args:
        state: Current graph TravelState.

    Returns:
        Partial state update containing `research_query`, `research_results`,
        `research_status`, and updated agent tracking.
    """
    logger.info("Executing research node")
    update = await run_research_agent(state)
    logger.info("Research node completed (research_status=%s)", update.get("research_status"))
    return update


async def validator_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Business Rule & Safety Validator Agent.

    Executes deterministic validation rules on extracted parameters, flight records,
    and research findings without altering raw underlying facts.

    Args:
        state: Current graph TravelState.

    Returns:
        Partial state update containing `validation_status`, `validation_errors`,
        and updated agent tracking.
    """
    logger.info("Executing validator node")
    update = await run_validator_agent(state)
    logger.info("Validator node completed (validation_status=%s)", update.get("validation_status"))
    return update


async def response_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Final Response Generation Agent.

    Synthesizes the validated operational state into a concise, factual,
    user-facing natural-language response.

    Args:
        state: Current graph TravelState.

    Returns:
        Partial state update containing `final_response`, `sources`,
        and updated agent tracking.
    """
    logger.info("Executing response node")
    update = await run_response_agent(state)
    logger.info("Response node completed")
    return update
