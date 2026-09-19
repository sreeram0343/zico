"""
Shared application workflow state module for ZICO.

This module defines `TravelState`, the central shared state structure passed
across LangGraph agent nodes, tool executors, business validators, and the
FastAPI presentation layer.

Architecture and Lifecycle:
    User Input -> Intent Detection -> Information Extraction ->
    Specialized Agent -> Tool Execution -> Validation -> Final Response

Design Principles:
    - Pure Data: Contains serializable Python primitives (str, int, float, bool, list, dict).
      No active runtime resources, clients, or connections.
    - Progressive Enrichment: Structured as a TypedDict to allow LangGraph nodes
      to return partial state updates without overwriting existing data.
    - Security & Privacy: No API keys, credentials, or private model reasoning
      chains are stored in state.
"""

from typing import Any, Dict, List, Optional, TypedDict


class FlightResult(TypedDict, total=False):
    """Normalized flight search result structure."""

    airline: Optional[str]
    flight_number: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    departure_time: Optional[str]
    arrival_time: Optional[str]
    price: Optional[float]
    currency: Optional[str]
    metadata: Optional[Dict[str, Any]]


class ResearchResult(TypedDict, total=False):
    """Structured research finding with source metadata."""

    title: str
    url: str
    content: str
    score: Optional[float]
    metadata: Optional[Dict[str, Any]]


class TravelState(TypedDict, total=False):
    """
    Authoritative shared state schema for ZICO travel workflow execution.

    Designed for LangGraph compatibility with total=False to support
    partial node updates across the multi-agent graph.
    """

    # -----------------------------------------------------------------------
    # A. Request / Session Information
    # -----------------------------------------------------------------------
    user_query: str
    session_id: str
    request_id: str

    # -----------------------------------------------------------------------
    # B. Intent Information
    # Initial supported categories: 'flight', 'research', 'general_travel', 'unsupported'
    # -----------------------------------------------------------------------
    intent: Optional[str]
    intent_confidence: Optional[float]

    # -----------------------------------------------------------------------
    # C. Travel Information (Extracted progressively, nullable if incomplete)
    # -----------------------------------------------------------------------
    origin: Optional[str]
    destination: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    passengers: Optional[int]

    # -----------------------------------------------------------------------
    # D. Flight-Related State
    # -----------------------------------------------------------------------
    flight_query: Optional[Dict[str, Any]]
    flight_results: List[Dict[str, Any]]
    # Status concepts: 'not_requested', 'pending', 'success', 'no_results', 'error'
    flight_status: Optional[str]

    # -----------------------------------------------------------------------
    # E. Research-Related State
    # -----------------------------------------------------------------------
    research_query: Optional[str]
    research_results: List[Dict[str, Any]]
    # Status concepts: 'not_requested', 'pending', 'success', 'no_results', 'error'
    research_status: Optional[str]

    # -----------------------------------------------------------------------
    # F. Agent Execution Information (Operational metadata, no chain-of-thought)
    # -----------------------------------------------------------------------
    active_agent: Optional[str]
    completed_agents: List[str]
    agent_messages: List[Dict[str, Any]]

    # -----------------------------------------------------------------------
    # G. Validation State
    # -----------------------------------------------------------------------
    # Status concepts: 'pending', 'passed', 'failed'
    validation_status: Optional[str]
    validation_errors: List[str]

    # -----------------------------------------------------------------------
    # H. Final Response State
    # -----------------------------------------------------------------------
    final_response: Optional[str]
    sources: List[Dict[str, Any]]

    # -----------------------------------------------------------------------
    # I. Workflow Status & Application Error Tracking
    # -----------------------------------------------------------------------
    # Status concepts: 'initialized', 'running', 'completed', 'failed'
    workflow_status: str
    # Application/tool errors (e.g. API timeout, network drop) kept separate from validation_errors
    errors: List[str]

    # -----------------------------------------------------------------------
    # J. Extensible Metadata
    # Operational metadata only (timestamps, provider IDs). Never store secrets or API keys.
    # -----------------------------------------------------------------------
    metadata: Dict[str, Any]


def create_initial_state(
    user_query: str,
    session_id: str,
    request_id: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> TravelState:
    """
    Initialize a clean, fully-isolated TravelState for a new workflow request.

    Guarantees that all collection fields are newly allocated instances to prevent
    shared mutable state between concurrent workflows or turns.

    Args:
        user_query: The incoming user message.
        session_id: Conversation session identifier.
        request_id: Request/workflow run identifier.
        metadata: Optional non-sensitive operational metadata.

    Returns:
        A valid, initialized TravelState dictionary.
    """
    return {
        # Request / Session
        "user_query": user_query,
        "session_id": session_id,
        "request_id": request_id,
        # Intent
        "intent": None,
        "intent_confidence": None,
        # Travel Parameters
        "origin": None,
        "destination": None,
        "departure_date": None,
        "return_date": None,
        "passengers": None,
        # Flight Operations
        "flight_query": None,
        "flight_results": [],
        "flight_status": "not_requested",
        # Research Operations
        "research_query": None,
        "research_results": [],
        "research_status": "not_requested",
        # Agent Execution
        "active_agent": None,
        "completed_agents": [],
        "agent_messages": [],
        # Validation
        "validation_status": "pending",
        "validation_errors": [],
        # Final Response
        "final_response": None,
        "sources": [],
        # Workflow Status & Errors
        "workflow_status": "initialized",
        "errors": [],
        # Extensible Metadata (Isolated copy)
        "metadata": dict(metadata) if metadata is not None else {},
    }
