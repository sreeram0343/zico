from datetime import datetime, timedelta
import re
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.disruption import DisruptionEvent, analyze_disruption, create_recovery_action
from app.graph.state import (
    ActionStatus,
    ActionType,
    PendingAction,
    TripConstraints,
    TripSegment,
    ZicoGraphState,
)
from app.graph.supervisor import supervisor_node
from app.graph.validators import detect_itinerary_conflicts, validate_budget_cap
from app.rag.service import get_rag_service
from app.tools.flight_search import search_flights

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

    normalized_messages: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, str):
            normalized_messages.append(HumanMessage(content=msg))
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if role == "assistant":
                normalized_messages.append(AIMessage(content=content))
            elif role == "system":
                normalized_messages.append(SystemMessage(content=content))
            else:
                normalized_messages.append(HumanMessage(content=content))
        elif isinstance(msg, BaseMessage):
            normalized_messages.append(msg)

    return {"messages": normalized_messages}


def flight_search_worker_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker specialized in querying SerpApi Google Flights, formatting results,
    and injecting structured flight options into the conversation.
    """
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])

    # Extract query text from latest message
    latest_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            latest_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Extract origin/destination patterns (e.g., JFK to LHR or SFO to ORD)
    origin, destination = "JFK", "LHR"
    iata_matches = re.findall(r"\b[A-Z]{3}\b", latest_text)
    if len(iata_matches) >= 2:
        origin, destination = iata_matches[0], iata_matches[1]
    elif "paris" in latest_text.lower():
        destination = "CDG"
    elif "tokyo" in latest_text.lower():
        destination = "HND"

    # Set search date (default 30 days in future)
    future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

    try:
        flight_results = search_flights.invoke({
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": future_date,
            "currency": "USD",
        })
    except Exception as exc:
        flight_results = [{"error": str(exc)}]

    # Format AI message response
    if flight_results and isinstance(flight_results, list) and len(flight_results) > 0 and isinstance(flight_results[0], TripSegment):
        options_summary = []
        for i, f in enumerate(flight_results[:3], 1):
            dep_time = f.start_time.strftime("%Y-%m-%d %H:%M")
            arr_time = f.end_time.strftime("%Y-%m-%d %H:%M")
            options_summary.append(
                f"{i}. **{f.title}** | {dep_time} -> {arr_time} | Price: {f.cost:.2f} {f.currency}"
            )
        response_text = (
            f"Here are the available flight options from **{origin}** to **{destination}** for {future_date}:\n\n"
            + "\n".join(options_summary)
            + "\n\nWould you like me to reserve one of these options into your itinerary?"
        )
    elif flight_results and isinstance(flight_results, list) and len(flight_results) > 0 and isinstance(flight_results[0], dict) and "error" not in flight_results[0]:
        options_summary = []
        for i, f in enumerate(flight_results[:3], 1):
            airline = f.get("airline", "Airline")
            f_num = f.get("flight_number", "")
            price = f.get("price", "N/A")
            curr = f.get("currency", "USD")
            dep_time = f.get("departure_time", "")
            arr_time = f.get("arrival_time", "")
            options_summary.append(
                f"{i}. **{airline}** ({f_num}) | {dep_time} -> {arr_time} | Price: {price} {curr}"
            )
        response_text = (
            f"Here are the available flight options from **{origin}** to **{destination}** for {future_date}:\n\n"
            + "\n".join(options_summary)
            + "\n\nWould you like me to reserve one of these options into your itinerary?"
        )
    else:
        err = flight_results[0].get("error", "Unable to retrieve flights") if flight_results and isinstance(flight_results[0], dict) else "No flights found"
        response_text = (
            f"I checked flight availability from **{origin}** to **{destination}**, but encountered: {err}. "
            "Please verify airport codes or search dates."
        )

    return {"messages": [AIMessage(content=response_text)]}



def policy_rag_worker_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker node retrieving airline policies, baggage limits, visa rules,
    and refund regulations from Qdrant vector store.
    """
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])

    latest_query = "travel policy guidelines"
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            latest_query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    rag_service = get_rag_service()
    rag_context = rag_service.format_rag_context(latest_query, limit=2)

    response_text = (
        f"Here are the verified travel policy regulations applicable to your inquiry:\n\n"
        f"{rag_context}\n\n"
        f"Let me know if you would like me to initiate a refund request, claim compensation, or review visa details."
    )

    return {"messages": [AIMessage(content=response_text)]}


def disruption_worker_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker specialized in evaluating delays, missed connections, cancellations,
    and formulating Human-in-the-Loop recovery actions.
    """
    if isinstance(state, dict):
        raw_itinerary = state.get("itinerary", [])
        raw_constraints = state.get("constraints", TripConstraints())
        existing_actions = state.get("pending_actions", [])
        existing_disruptions = state.get("active_disruptions", [])
    else:
        raw_itinerary = getattr(state, "itinerary", [])
        raw_constraints = getattr(state, "constraints", TripConstraints())
        existing_actions = getattr(state, "pending_actions", [])
        existing_disruptions = getattr(state, "active_disruptions", [])

    itinerary = [
        s if isinstance(s, TripSegment) else TripSegment.model_validate(s)
        for s in raw_itinerary
    ]
    constraints = (
        raw_constraints
        if isinstance(raw_constraints, TripConstraints)
        else TripConstraints.model_validate(raw_constraints)
    )

    # Detect delay or cancellation target from first segment or active disruption
    new_actions = list(existing_actions)
    new_disruptions = list(existing_disruptions)

    if itinerary:
        target_seg = itinerary[0]
        event = DisruptionEvent(
            segment_id=target_seg.id,
            event_type="DELAY",
            delay_minutes=60,
            reason="Inbound aircraft maintenance delay",
        )
        impact = analyze_disruption(itinerary, event, constraints)
        new_disruptions.append({
            "type": "DISRUPTION_IMPACT",
            "affected_segment_id": impact.affected_segment_id,
            "severity": impact.severity,
            "summary": impact.summary,
        })

        action = create_recovery_action(itinerary, impact, event, constraints)
        if not action:
            import uuid
            action = PendingAction(
                action_id=f"act_{uuid.uuid4().hex[:8]}",
                action_type=ActionType.RESCHEDULE,
                description=f"Schedule adjustment advisory for '{target_seg.title}' due to {event.delay_minutes}m delay.",
                payload={"disruption_type": "DELAY", "affected_segment_id": target_seg.id},
                requires_explicit_approval=True,
                status=ActionStatus.PENDING,
            )

        new_actions.append(action)
        response_text = (
            f"⚠️ **Travel Disruption Advisory**:\n{impact.summary}\n\n"
            f"I have prepared an action proposal (**{action.action_id}**): {action.description}. "
            f"Please review and confirm to proceed."
        )

    else:
        response_text = "No active trip segments found in current itinerary to evaluate disruptions."

    return {
        "pending_actions": new_actions,
        "active_disruptions": new_disruptions,
        "messages": [AIMessage(content=response_text)],
    }


def validator_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes deterministic itinerary conflict detection and updates active disruptions.
    """
    if isinstance(state, dict):
        raw_itinerary = state.get("itinerary", [])
        raw_constraints = state.get("constraints", TripConstraints())
        existing_disruptions = state.get("active_disruptions", [])
    else:
        raw_itinerary = getattr(state, "itinerary", [])
        raw_constraints = getattr(state, "constraints", TripConstraints())
        existing_disruptions = getattr(state, "active_disruptions", [])

    itinerary: List[TripSegment] = []
    for seg in raw_itinerary:
        if isinstance(seg, TripSegment):
            itinerary.append(seg)
        elif isinstance(seg, dict):
            itinerary.append(TripSegment.model_validate(seg))

    if isinstance(raw_constraints, dict):
        constraints = TripConstraints.model_validate(raw_constraints)
    elif isinstance(raw_constraints, TripConstraints):
        constraints = raw_constraints
    else:
        constraints = TripConstraints()

    # 1. Detect temporal conflicts and connection buffer deficits
    conflicts = detect_itinerary_conflicts(itinerary, constraints)

    disruptions: List[Dict[str, Any]] = [
        d for d in existing_disruptions if d.get("type") not in ("ITINERARY_CONFLICT", "BUDGET_EXCEEDED")
    ]

    for c in conflicts:
        disruptions.append({
            "type": "ITINERARY_CONFLICT",
            "segment_a_id": c.segment_a_id,
            "segment_b_id": c.segment_b_id,
            "reason": c.reason,
            "deficit_minutes": c.deficit_minutes,
        })

    # 2. Validate budget cap
    if constraints.max_budget is not None and not validate_budget_cap(itinerary, constraints.max_budget):
        total_cost = sum(seg.cost for seg in itinerary)
        disruptions.append({
            "type": "BUDGET_EXCEEDED",
            "max_budget": constraints.max_budget,
            "current_total": total_cost,
            "reason": f"Total itinerary cost ({total_cost}) exceeds budget cap ({constraints.max_budget})",
        })

    return {"active_disruptions": disruptions}


# ---------------------------------------------------------------------------
# Router Conditional Function
# ---------------------------------------------------------------------------


def supervisor_router(state: ZicoGraphState | Dict[str, Any]) -> str:
    """
    Evaluates next_node set by supervisor_node and routes to appropriate branch.
    """
    if isinstance(state, dict):
        next_node = state.get("next_node")
    else:
        next_node = getattr(state, "next_node", None)

    valid_destinations = {
        "flight_search_worker",
        "policy_rag_worker",
        "disruption_worker",
        "validator_node",
    }

    if next_node in valid_destinations:
        return next_node
    return "validator_node"


# ---------------------------------------------------------------------------
# StateGraph Construction
# ---------------------------------------------------------------------------


def build_zico_graph() -> StateGraph:
    """Builds and wires the full multi-agent ZICO LangGraph state machine."""
    graph_builder = StateGraph(ZicoGraphState)

    # Register Nodes
    graph_builder.add_node("input_node", input_node)
    graph_builder.add_node("supervisor_node", supervisor_node)
    graph_builder.add_node("flight_search_worker", flight_search_worker_node)
    graph_builder.add_node("policy_rag_worker", policy_rag_worker_node)
    graph_builder.add_node("disruption_worker", disruption_worker_node)
    graph_builder.add_node("validator_node", validator_node)

    # Initial flow
    graph_builder.add_edge(START, "input_node")
    graph_builder.add_edge("input_node", "supervisor_node")

    # Conditional Routing from Supervisor
    graph_builder.add_conditional_edges(
        "supervisor_node",
        supervisor_router,
        {
            "flight_search_worker": "flight_search_worker",
            "policy_rag_worker": "policy_rag_worker",
            "disruption_worker": "disruption_worker",
            "validator_node": "validator_node",
        },
    )

    # Worker flows converge on deterministic validator
    graph_builder.add_edge("flight_search_worker", "validator_node")
    graph_builder.add_edge("policy_rag_worker", "validator_node")
    graph_builder.add_edge("disruption_worker", "validator_node")
    graph_builder.add_edge("validator_node", END)

    return graph_builder


def create_zico_graph(checkpointer: Optional[Any] = None):
    """Compiles the ZICO graph with MemorySaver or external checkpointer."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    builder = build_zico_graph()
    return builder.compile(checkpointer=checkpointer)


# Global engine instance compiled with MemorySaver checkpointer
default_checkpointer = MemorySaver()
graph_engine = create_zico_graph(checkpointer=default_checkpointer)
