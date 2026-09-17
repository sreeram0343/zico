"""
Complete Executable LangGraph Workflow for ZICO Travel Operations.

This module assembles the ZICO multi-agent travel operations graph, connecting:
    - Router Agent: Intent classification (entry point).
    - Flight Agent: Flight tracking, schedules, and status via AviationStack.
    - Research Agent: Live policy, regulation, and advisory research via Tavily.
    - Validator Agent: Business rule and safety validation on retrieved facts.
    - Response Agent: Fact-grounded, validation-aware final response generation.

Workflow Topology:
    START
      │
      ▼
    router
      │
      ├─[flight]─────────► flight ──────┐
      ├─[research]───────► research ────┤
      ├─[general_travel]─► research ────┤
      │                                 ▼
      │                             validator
      │                                 │
      ├─[unsupported]───────────────────┤
      └─[invalid/missing]───────────────┤
                                        ▼
                                     response
                                        │
                                        ▼
                                       END

Architectural Principles:
    - Pure Orchestration: Connects existing nodes and edges without duplicating
      agent business logic or routing decisions.
    - Strict Validation Invariant: Specialized paths (flight, research) always pass
      through the Validator before reaching the Response Agent.
    - Direct Unsupported Termination: Out-of-scope inquiries bypass specialized tools
      and validation, routing directly to Response for explanation.
    - Pure StateGraph: Operates exclusively on `TravelState` from `app.core.state`.
    - Thread & Concurrency Safe: Exposes compiled graph with isolated execution state.
    - Zero External Network on Import/Build: No external APIs or LLMs are contacted
      during graph construction or compilation.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.logging import get_logger
from app.core.state import TravelState
from app.graph.edges import route_after_router
from app.graph.nodes import (
    flight_node,
    research_node,
    response_node,
    router_node,
    validator_node,
)

logger = get_logger(__name__)

__all__ = [
    "create_workflow",
    "build_workflow",
    "zico_graph",
    "run_workflow",
]


def create_workflow() -> StateGraph:
    """
    Construct the uncompiled ZICO StateGraph workflow definition.

    Registers all operational nodes, connects START to the router node,
    configures conditional routing after the router, enforces the validator
    stage for specialized paths, and terminates after response generation.

    Returns:
        Fresh, uncompiled StateGraph instance parameterized with TravelState.
    """
    logger.info("Building ZICO workflow StateGraph")

    builder = StateGraph(TravelState)

    # 1. Register Core Nodes
    builder.add_node("router", router_node)
    builder.add_node("flight", flight_node)
    builder.add_node("research", research_node)
    builder.add_node("validator", validator_node)
    builder.add_node("response", response_node)

    # 2. Set Workflow Entry Point: START -> router
    builder.add_edge(START, "router")

    # 3. Conditional Routing after Intent Router
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "flight": "flight",
            "research": "research",
            "response": "response",
        },
    )

    # 4. Specialized Agent Paths Always Flow to Validator
    builder.add_edge("flight", "validator")
    builder.add_edge("research", "validator")

    # 5. Validator Always Flows to Response
    builder.add_edge("validator", "response")

    # 6. Response Node Terminates the Workflow: response -> END
    builder.add_edge("response", END)

    return builder


def build_workflow() -> CompiledStateGraph:
    """
    Construct and compile the executable ZICO LangGraph workflow.

    Returns:
        CompiledStateGraph instance ready for async or sync execution.
    """
    builder = create_workflow()
    compiled = builder.compile()
    logger.info("ZICO workflow compiled successfully")
    return compiled


# Module-level singleton compiled workflow graph
zico_graph: CompiledStateGraph = build_workflow()


async def run_workflow(
    state: TravelState,
    graph: Optional[CompiledStateGraph] = None,
) -> TravelState:
    """
    Execute the ZICO LangGraph workflow asynchronously on the given TravelState.

    Args:
        state: Initial or active TravelState dictionary.
        graph: Optional compiled graph override (useful for testing). Defaults to `zico_graph`.

    Returns:
        Updated TravelState after complete graph execution to END.
    """
    active_graph = graph if graph is not None else zico_graph
    result = await active_graph.ainvoke(state)
    return result
