"""
LangGraph Node Adapter Layer for ZICO Travel Operations.

This module provides thin, asynchronous LangGraph-compatible node functions
wrapping ZICO's core domain agents with structured lifecycle tracing:
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
    - Lifecycle Tracing: Automatically records agent_started, agent_completed, and
      agent_failed structured events with duration_ms, request_id, and session_id.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from app.agents.flight_agent import run_flight_agent
from app.agents.research_agent import run_research_agent
from app.agents.response_agent import run_response_agent
from app.agents.router_agent import aroute_request
from app.agents.validator_agent import run_validator_agent
from app.core.logging import get_logger
from app.core.request_context import get_request_id, get_session_id
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
    """
    agent_name = "router_agent"
    req_id = get_request_id() or state.get("request_id") or "-"
    sess_id = get_session_id() or state.get("session_id") or "-"

    start_time = time.perf_counter()
    logger.info(
        "agent_started agent=%s request_id=%s session_id=%s",
        agent_name,
        req_id,
        sess_id,
    )
    try:
        update = await aroute_request(state)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "agent_completed agent=%s request_id=%s session_id=%s duration_ms=%.2f",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
        )
        return update
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        exc_type = type(exc).__name__
        logger.error(
            "agent_failed agent=%s request_id=%s session_id=%s duration_ms=%.2f error_type=%s",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
            exc_type,
        )
        raise


async def flight_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Flight Operations Agent.

    Extracts flight parameters, resolves airport/city locations, and queries
    AviationStack flight data, writing normalized flight results to state.
    """
    agent_name = "flight_agent"
    req_id = get_request_id() or state.get("request_id") or "-"
    sess_id = get_session_id() or state.get("session_id") or "-"

    start_time = time.perf_counter()
    logger.info(
        "agent_started agent=%s request_id=%s session_id=%s",
        agent_name,
        req_id,
        sess_id,
    )
    try:
        update = await run_flight_agent(state)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "agent_completed agent=%s request_id=%s session_id=%s duration_ms=%.2f",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
        )
        return update
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        exc_type = type(exc).__name__
        logger.error(
            "agent_failed agent=%s request_id=%s session_id=%s duration_ms=%.2f error_type=%s",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
            exc_type,
        )
        raise


async def research_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Travel Research Agent.

    Generates optimized search queries and queries Tavily search API,
    storing deduplicated, normalized research findings into state.
    """
    agent_name = "research_agent"
    req_id = get_request_id() or state.get("request_id") or "-"
    sess_id = get_session_id() or state.get("session_id") or "-"

    start_time = time.perf_counter()
    logger.info(
        "agent_started agent=%s request_id=%s session_id=%s",
        agent_name,
        req_id,
        sess_id,
    )
    try:
        update = await run_research_agent(state)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "agent_completed agent=%s request_id=%s session_id=%s duration_ms=%.2f",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
        )
        return update
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        exc_type = type(exc).__name__
        logger.error(
            "agent_failed agent=%s request_id=%s session_id=%s duration_ms=%.2f error_type=%s",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
            exc_type,
        )
        raise


async def validator_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Business Rule & Safety Validator Agent.

    Executes deterministic validation rules on extracted parameters, flight records,
    and research findings without altering raw underlying facts.
    """
    agent_name = "validator_agent"
    req_id = get_request_id() or state.get("request_id") or "-"
    sess_id = get_session_id() or state.get("session_id") or "-"

    start_time = time.perf_counter()
    logger.info(
        "agent_started agent=%s request_id=%s session_id=%s",
        agent_name,
        req_id,
        sess_id,
    )
    try:
        update = await run_validator_agent(state)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "agent_completed agent=%s request_id=%s session_id=%s duration_ms=%.2f",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
        )
        return update
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        exc_type = type(exc).__name__
        logger.error(
            "agent_failed agent=%s request_id=%s session_id=%s duration_ms=%.2f error_type=%s",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
            exc_type,
        )
        raise


async def response_node(state: TravelState) -> Dict[str, Any]:
    """
    LangGraph node adapter for the Final Response Generation Agent.

    Synthesizes the validated operational state into a concise, factual,
    user-facing natural-language response.
    """
    agent_name = "response_agent"
    req_id = get_request_id() or state.get("request_id") or "-"
    sess_id = get_session_id() or state.get("session_id") or "-"

    start_time = time.perf_counter()
    logger.info(
        "agent_started agent=%s request_id=%s session_id=%s",
        agent_name,
        req_id,
        sess_id,
    )
    try:
        update = await run_response_agent(state)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "agent_completed agent=%s request_id=%s session_id=%s duration_ms=%.2f",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
        )
        return update
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        exc_type = type(exc).__name__
        logger.error(
            "agent_failed agent=%s request_id=%s session_id=%s duration_ms=%.2f error_type=%s",
            agent_name,
            req_id,
            sess_id,
            duration_ms,
            exc_type,
        )
        raise
