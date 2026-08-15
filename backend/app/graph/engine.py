from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from app.graph.state import (
    TripConstraints,
    TripSegment,
    ZicoGraphState,
)
from app.graph.validators import detect_itinerary_conflicts, validate_budget_cap

# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------


def input_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses, validates, and normalizes incoming message state.
    """
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])

    # If incoming messages are plain strings or dicts, wrap in HumanMessage
    normalized_messages: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, str):
            normalized_messages.append(HumanMessage(content=msg))
        elif isinstance(msg, dict):
            normalized_messages.append(
                HumanMessage(content=msg.get("content", ""))
            )
        elif isinstance(msg, BaseMessage):
            normalized_messages.append(msg)

    return {"messages": normalized_messages}


def supervisor_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder supervisor routing node that directs flow through the state machine.
    """
    # For Phase 1 foundational flow, defaults routing to validator_node
    return {"next_node": "validator_node"}


def validator_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes deterministic itinerary conflict detection and updates active disruptions.
    """
    if isinstance(state, dict):
        raw_itinerary = state.get("itinerary", [])
        raw_constraints = state.get("constraints", TripConstraints())
    else:
        raw_itinerary = getattr(state, "itinerary", [])
        raw_constraints = getattr(state, "constraints", TripConstraints())

    # Ensure segments are TripSegment instances
    itinerary: List[TripSegment] = []
    for seg in raw_itinerary:
        if isinstance(seg, TripSegment):
            itinerary.append(seg)
        elif isinstance(seg, dict):
            itinerary.append(TripSegment.model_validate(seg))

    # Ensure constraints is a TripConstraints instance
    if isinstance(raw_constraints, dict):
        constraints = TripConstraints.model_validate(raw_constraints)
    elif isinstance(raw_constraints, TripConstraints):
        constraints = raw_constraints
    else:
        constraints = TripConstraints()

    # 1. Detect temporal conflicts and connection buffer deficits
    conflicts = detect_itinerary_conflicts(itinerary, constraints)

    disruptions: List[Dict[str, Any]] = [
        {
            "type": "ITINERARY_CONFLICT",
            "segment_a_id": c.segment_a_id,
            "segment_b_id": c.segment_b_id,
            "reason": c.reason,
            "deficit_minutes": c.deficit_minutes,
        }
        for c in conflicts
    ]

    # 2. Validate budget cap
    if constraints.max_budget is not None and not validate_budget_cap(
        itinerary, constraints.max_budget
    ):
        total_cost = sum(seg.cost for seg in itinerary)
        disruptions.append(
            {
                "type": "BUDGET_EXCEEDED",
                "max_budget": constraints.max_budget,
                "current_total": total_cost,
                "reason": (
                    f"Total itinerary cost ({total_cost}) exceeds budget cap "
                    f"({constraints.max_budget})"
                ),
            }
        )

    return {"active_disruptions": disruptions}


# ---------------------------------------------------------------------------
# StateGraph Construction
# ---------------------------------------------------------------------------


def build_zico_graph() -> StateGraph:
    """Builds and wires the foundational Zico LangGraph state graph."""
    graph_builder = StateGraph(ZicoGraphState)

    # Add foundational nodes
    graph_builder.add_node("input_node", input_node)
    graph_builder.add_node("supervisor_node", supervisor_node)
    graph_builder.add_node("validator_node", validator_node)

    # Add linear flow edges
    graph_builder.add_edge(START, "input_node")
    graph_builder.add_edge("input_node", "supervisor_node")
    graph_builder.add_edge("supervisor_node", "validator_node")
    graph_builder.add_edge("validator_node", END)

    return graph_builder


def create_zico_graph(checkpointer: Optional[Any] = None):
    """Compiles the Zico graph with MemorySaver or provided checkpointer."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    builder = build_zico_graph()
    return builder.compile(checkpointer=checkpointer)


# Global engine instance compiled with MemorySaver checkpointer
default_checkpointer = MemorySaver()
graph_engine = create_zico_graph(checkpointer=default_checkpointer)
