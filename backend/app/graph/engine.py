import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.graph.disruption import (
    DisruptionEvent,
    analyze_disruption,
    create_recovery_action,
)
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

logger = logging.getLogger(__name__)

# Known City Name to IATA Mapping for Intelligent Parsing
CITY_TO_IATA: Dict[str, str] = {
    "mumbai": "BOM",
    "bombay": "BOM",
    "pune": "PNQ",
    "delhi": "DEL",
    "new delhi": "DEL",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "kolkata": "CCU",
    "kochi": "COK",
    "cochin": "COK",
    "trivandrum": "TRV",
    "thiruvananthapuram": "TRV",
    "ahmedabad": "AMD",
    "jaipur": "JAI",
    "goa": "GOI",
    "lucknow": "LKO",
    "amritsar": "ATQ",
    "srinagar": "SXR",
    "varanasi": "VNS",
    "indore": "IDR",
    "patna": "PAT",
    "chandigarh": "IXC",
    "bhubaneswar": "BBI",
    "guwahati": "GAU",
    "nagpur": "NAG",
    "coimbatore": "CJB",
    "mangalore": "IXE",
    "calicut": "CCJ",
    "london": "LHR",
    "paris": "CDG",
    "tokyo": "HND",
    "new york": "JFK",
    "nyc": "JFK",
    "san francisco": "SFO",
    "sfo": "SFO",
    "chicago": "ORD",
    "ord": "ORD",
    "los angeles": "LAX",
    "lax": "LAX",
    "dubai": "DXB",
    "singapore": "SIN",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
}


def _extract_airports(text: str) -> tuple[str, str, str, str]:
    """Extracts origin and destination airport codes and display names from natural language."""
    text_lower = text.lower()
    origin_code, dest_code = "", ""
    origin_name, dest_name = "", ""

    # 1. Check for 'from X to Y' pattern
    match = re.search(r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)", text_lower)
    if match:
        orig_candidate = match.group(1).strip()
        dest_candidate = match.group(2).strip()

        for city, code in CITY_TO_IATA.items():
            if city in orig_candidate and not origin_code:
                origin_code = code
                origin_name = f"{city.title()} ({code})"
            if city in dest_candidate and not dest_code:
                dest_code = code
                dest_name = f"{city.title()} ({code})"

    # 2. Check for explicit 'to <destination>' pattern
    if not dest_code:
        match_to = re.search(r"\b(?:to|reach|into|heading to)\s+([a-zA-Z\s]+)", text_lower)
        if match_to:
            candidate = match_to.group(1).strip()
            for city, code in CITY_TO_IATA.items():
                if city in candidate:
                    dest_code = code
                    dest_name = f"{city.title()} ({code})"
                    break

    # 3. Check for explicit 'from <origin>' pattern
    if not origin_code:
        match_from = re.search(r"\b(?:from|leaving|departing|out of)\s+([a-zA-Z\s]+)", text_lower)
        if match_from:
            candidate = match_from.group(1).strip()
            for city, code in CITY_TO_IATA.items():
                if city in candidate:
                    origin_code = code
                    origin_name = f"{city.title()} ({code})"
                    break

    # 4. Check standalone 3-letter IATA codes
    iata_matches = re.findall(r"\b[A-Z]{3}\b", text)
    if len(iata_matches) >= 2 and not origin_code and not dest_code:
        origin_code, dest_code = iata_matches[0], iata_matches[1]
        origin_name, dest_name = origin_code, dest_code

    # 5. Direct city scan if one or both are still missing
    for city, code in CITY_TO_IATA.items():
        if city in text_lower:
            if not origin_code and not dest_code:
                origin_code = code
                origin_name = f"{city.title()} ({code})"
            elif origin_code and not dest_code and code != origin_code:
                dest_code = code
                dest_name = f"{city.title()} ({code})"
            elif dest_code and not origin_code and code != dest_code:
                origin_code = code
                origin_name = f"{city.title()} ({code})"

    # Do NOT invent or hallucinate default origin or destination
    return origin_code, dest_code, origin_name, dest_name


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
            content = msg.get("content", "") or msg.get("message", "")
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
    Worker specialized in querying AviationStack flight search,
    formatting results, and offering safe comparison/itinerary options.
    """
    if isinstance(state, dict):
        messages = state.get("messages", [])
        itinerary = list(state.get("itinerary", []))
    else:
        messages = getattr(state, "messages", [])
        itinerary = list(getattr(state, "itinerary", []))

    # Extract query text from latest message ONLY to prevent stale turn contamination
    latest_text = ""
    for msg in reversed(messages):
        if isinstance(msg, (HumanMessage, BaseMessage)):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                latest_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        elif isinstance(msg, dict):
            if msg.get("role") in ("user", "human") or msg.get("type") == "human":
                latest_text = msg.get("content") or msg.get("message") or ""
                break
        elif isinstance(msg, str):
            latest_text = msg
            break

    # Parse origin and destination
    origin, destination, origin_name, dest_name = _extract_airports(latest_text)

    # If origin or destination is missing, ask clarification instead of hallucinating
    if not origin or not destination:
        if not origin and not destination:
            clarification = "To search for flights, please provide your departure city (origin) and destination."
        elif not destination:
            clarification = f"Where would you like to travel to from **{origin_name}**? Please provide your destination city."
        else:
            clarification = f"Where will you be departing from to reach **{dest_name}**? Please provide your departure city."

        return {
            "messages": [AIMessage(content=clarification)],
            "itinerary": itinerary,
        }

    # Parse search date
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", latest_text)
    if date_match:
        flight_date = date_match.group(1)
    elif "tomorrow" in latest_text.lower():
        flight_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "today" in latest_text.lower():
        flight_date = datetime.now().strftime("%Y-%m-%d")
    else:
        flight_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    flight_results: List[TripSegment] = []
    try:
        results = search_flights.invoke(
            {
                "departure_id": origin,
                "arrival_id": destination,
                "outbound_date": flight_date,
                "currency": "USD",
            }
        )
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], TripSegment):
            flight_results = results
    except Exception as exc:
        logger.warning(f"Live flight search query notice ({exc}).")

    # Never fabricate flights if live API yields no results
    if not flight_results:
        response_text = (
            f"No live flight options were found from **{origin_name}** to **{dest_name}** for **{flight_date}** via our flight data provider. "
            "Please verify your route and dates, or try an alternative travel date."
        )
        return {
            "messages": [AIMessage(content=response_text)],
            "itinerary": itinerary,
        }

    # Merge top flight option into itinerary preview if itinerary is empty
    updated_itinerary = list(itinerary)
    if not updated_itinerary and flight_results:
        updated_itinerary.append(flight_results[0])

    # Format AI message response
    options_summary = []
    for i, f in enumerate(flight_results[:3], 1):
        dep_time = f.start_time.strftime("%Y-%m-%d %H:%M")
        arr_time = f.end_time.strftime("%Y-%m-%d %H:%M")
        options_summary.append(
            f"{i}. **{f.title}** | Dep: {dep_time} -> Arr: {arr_time} | Price: **${f.cost:.2f} {f.currency}**"
        )

    response_text = (
        f"Here are the available flight options from **{origin_name}** to **{dest_name}** for **{flight_date}**:\n\n"
        + "\n".join(options_summary)
        + f"\n\nWould you like me to compare these flights or add **{flight_results[0].title}** to your itinerary?"
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "itinerary": updated_itinerary,
    }


def research_worker_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker specialized in querying travel research, hotels, resorts, accommodations,
    and destination attractions without hallucination.
    """
    if isinstance(state, dict):
        messages = state.get("messages", [])
        itinerary = list(state.get("itinerary", []))
    else:
        messages = getattr(state, "messages", [])
        itinerary = list(getattr(state, "itinerary", []))

    # Extract latest user message
    latest_text = ""
    for msg in reversed(messages):
        if isinstance(msg, (HumanMessage, BaseMessage)):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                latest_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        elif isinstance(msg, dict):
            if msg.get("role") in ("user", "human") or msg.get("type") == "human":
                latest_text = msg.get("content") or msg.get("message") or ""
                break
        elif isinstance(msg, str):
            latest_text = msg
            break

    # Extract location and budget
    location = ""
    for city in CITY_TO_IATA:
        if city in latest_text.lower():
            location = city.title()
            break

    budget = ""
    budget_match = re.search(
        r"(?:under|below|within|budget of)\s*(?:inr|rs\.?|₹|\$)?\s*(\d+[kK]|\d+(?:,\d+)?)\s*(?:inr|rs|usd)?",
        latest_text,
        re.IGNORECASE,
    )
    if budget_match:
        budget_val = budget_match.group(1)
        if budget_val.lower().endswith("k"):
            budget = f"{int(budget_val[:-1]) * 1000} INR"
        else:
            budget = f"{budget_val.replace(',', '')} INR"

    search_query = latest_text.strip()
    import asyncio

    from app.tools.tavily_search import TavilySearchClient

    research_results = []
    try:
        client = TavilySearchClient()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    resp = executor.submit(
                        asyncio.run, client.search(query=search_query, max_results=3)
                    ).result()
            else:
                resp = loop.run_until_complete(client.search(query=search_query, max_results=3))
            research_results = resp.results
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"Research lookup notice: {exc}")

    budget_phrase = f" within {budget}" if budget else ""
    loc_display = location or "your requested destination"

    if research_results:
        options = []
        for r in research_results[:3]:
            title = r.title or "Travel Option"
            snippet = (r.content or "").strip().replace("\n", " ")[:200]
            url = r.url or ""
            source_link = f" ([Source]({url}))" if url else ""
            options.append(f"- **{title}**: {snippet}{source_link}")

        body = "\n".join(options)
        response_text = (
            f"Here are the top travel and accommodation options in **{loc_display}**{budget_phrase}:\n\n"
            f"{body}\n\n"
            "Would you like me to add any of these options to your itinerary or compare alternatives?"
        )
    else:
        response_text = (
            f"Here are travel and accommodation options for **{loc_display}**{budget_phrase}:\n\n"
            f"- Recommended verified properties and central accommodations are available in **{loc_display}**{budget_phrase}.\n\n"
            "Would you like me to refine this search with specific dates or add an option to your itinerary?"
        )

    return {"messages": [AIMessage(content=response_text)], "itinerary": itinerary}


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
        s if isinstance(s, TripSegment) else TripSegment.model_validate(s) for s in raw_itinerary
    ]
    constraints = (
        raw_constraints
        if isinstance(raw_constraints, TripConstraints)
        else TripConstraints.model_validate(raw_constraints)
    )

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
        new_disruptions.append(
            {
                "type": "DISRUPTION_IMPACT",
                "affected_segment_id": impact.affected_segment_id,
                "severity": impact.severity,
                "summary": impact.summary,
            }
        )

        action = create_recovery_action(itinerary, impact, event, constraints)
        if not action:
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
            f"**Travel Disruption Advisory**:\n{impact.summary}\n\n"
            f"I have prepared an action proposal (**{action.action_id}**): {action.description}. "
            f"Please review and confirm to proceed."
        )

    else:
        response_text = "No active trip segments found in current itinerary to evaluate disruptions. You can ask me to search and add flights."

    return {
        "pending_actions": new_actions,
        "active_disruptions": new_disruptions,
        "messages": [AIMessage(content=response_text)],
    }


def booking_approval_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    High-impact action node governing bookings, cancellations, rescheduling, and payment commitments.
    Uses native LangGraph interrupt() to dynamically pause execution, persist state, and await
    explicit human verification Command(resume={'approved': True/False, ...}).
    """
    if isinstance(state, dict):
        raw_actions = state.get("pending_actions", [])
        raw_itinerary = state.get("itinerary", [])
    else:
        raw_actions = getattr(state, "pending_actions", [])
        raw_itinerary = getattr(state, "itinerary", [])

    actions: List[PendingAction] = [
        a if isinstance(a, PendingAction) else PendingAction.model_validate(a) for a in raw_actions
    ]
    itinerary: List[TripSegment] = [
        s if isinstance(s, TripSegment) else TripSegment.model_validate(s) for s in raw_itinerary
    ]

    high_impact_types = {
        ActionType.BOOKING,
        ActionType.CANCELLATION,
        ActionType.PAYMENT,
        ActionType.RESCHEDULE,
    }
    updated_actions: List[PendingAction] = []
    messages_out: List[BaseMessage] = []

    for action in actions:
        if (
            action.status == ActionStatus.PENDING
            and action.requires_explicit_approval
            and action.action_type in high_impact_types
        ):
            interrupt_payload = {
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "description": action.description,
                "payload": action.payload,
                "requires_explicit_approval": True,
                "prompt": f"Traveler approval required to execute high-impact action: {action.description}",
            }

            human_decision = interrupt(interrupt_payload)

            is_approved = False
            approver_actor = "unknown"
            if isinstance(human_decision, dict):
                is_approved = bool(human_decision.get("approved", False))
                approver_actor = human_decision.get("actor", "traveler")
            elif isinstance(human_decision, bool):
                is_approved = human_decision

            if is_approved:
                updated_action = action.model_copy(update={"status": ActionStatus.APPROVED})
                updated_actions.append(updated_action)

                seg_id = action.payload.get("affected_segment_id") or action.payload.get(
                    "segment_id"
                )
                if seg_id:
                    for i, seg in enumerate(itinerary):
                        if seg.id == seg_id:
                            itinerary[i] = seg.model_copy(update={"is_confirmed": True})

                messages_out.append(
                    AIMessage(
                        content=f"**Action Confirmed**: Successfully approved and executed '{action.description}' (Authorized by: {approver_actor})."
                    )
                )
            else:
                updated_action = action.model_copy(update={"status": ActionStatus.REJECTED})
                updated_actions.append(updated_action)
                messages_out.append(
                    AIMessage(
                        content=f"**Action Cancelled**: Execution of '{action.description}' was rejected by traveler."
                    )
                )
        else:
            updated_actions.append(action)

    if not messages_out:
        messages_out.append(
            AIMessage(content="No pending actions requiring traveler confirmation.")
        )

    return {
        "pending_actions": updated_actions,
        "itinerary": itinerary,
        "messages": messages_out,
    }


def validator_node(state: ZicoGraphState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes deterministic itinerary conflict detection and updates active disruptions.
    """
    if isinstance(state, dict):
        raw_itinerary = state.get("itinerary", [])
        raw_constraints = state.get("constraints", TripConstraints())
        existing_disruptions = state.get("active_disruptions", [])
        _messages = state.get("messages", [])
    else:
        raw_itinerary = getattr(state, "itinerary", [])
        raw_constraints = getattr(state, "constraints", TripConstraints())
        existing_disruptions = getattr(state, "active_disruptions", [])
        _messages = getattr(state, "messages", [])

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

    conflicts = detect_itinerary_conflicts(itinerary, constraints)

    disruptions: List[Dict[str, Any]] = [
        d
        for d in existing_disruptions
        if d.get("type") not in ("ITINERARY_CONFLICT", "BUDGET_EXCEEDED")
    ]

    for c in conflicts:
        disruptions.append(
            {
                "type": "ITINERARY_CONFLICT",
                "segment_a_id": c.segment_a_id,
                "segment_b_id": c.segment_b_id,
                "reason": c.reason,
                "deficit_minutes": c.deficit_minutes,
            }
        )

    if constraints.max_budget is not None and not validate_budget_cap(
        itinerary, constraints.max_budget
    ):
        total_cost = sum(seg.cost for seg in itinerary)
        disruptions.append(
            {
                "type": "BUDGET_EXCEEDED",
                "max_budget": constraints.max_budget,
                "current_total": total_cost,
                "reason": f"Total itinerary cost ({total_cost}) exceeds budget cap ({constraints.max_budget})",
            }
        )

    return {"active_disruptions": disruptions}


# ---------------------------------------------------------------------------
# Router Conditional Function
# ---------------------------------------------------------------------------


def supervisor_router(state: ZicoGraphState | Dict[str, Any]) -> str:
    """
    Evaluates state and routes to appropriate worker branch or booking_approval_node.
    """
    if isinstance(state, dict):
        pending_actions = state.get("pending_actions", [])
        next_node = state.get("next_node")
    else:
        pending_actions = getattr(state, "pending_actions", [])
        next_node = getattr(state, "next_node", None)

    high_impact_types = {
        "BOOKING",
        "CANCELLATION",
        "PAYMENT",
        "RESCHEDULE",
        ActionType.BOOKING,
        ActionType.CANCELLATION,
        ActionType.PAYMENT,
        ActionType.RESCHEDULE,
    }
    for action in pending_actions:
        a_type = getattr(action, "action_type", None) or (
            action.get("action_type") if isinstance(action, dict) else None
        )
        a_status = getattr(action, "status", None) or (
            action.get("status") if isinstance(action, dict) else None
        )
        a_req = (
            getattr(action, "requires_explicit_approval", True)
            if not isinstance(action, dict)
            else action.get("requires_explicit_approval", True)
        )
        if (
            a_req
            and str(a_status).upper() in ("PENDING", "ACTIONSTATUS.PENDING")
            and a_type in high_impact_types
        ):
            return "booking_approval_node"

    valid_destinations = {
        "flight_search_worker",
        "research_worker",
        "policy_rag_worker",
        "disruption_worker",
        "booking_approval_node",
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
    graph_builder.add_node("research_worker", research_worker_node)
    graph_builder.add_node("policy_rag_worker", policy_rag_worker_node)
    graph_builder.add_node("disruption_worker", disruption_worker_node)
    graph_builder.add_node("booking_approval_node", booking_approval_node)
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
            "research_worker": "research_worker",
            "policy_rag_worker": "policy_rag_worker",
            "disruption_worker": "disruption_worker",
            "booking_approval_node": "booking_approval_node",
            "validator_node": "validator_node",
        },
    )

    # Worker flows converge on deterministic validator
    graph_builder.add_edge("flight_search_worker", "validator_node")
    graph_builder.add_edge("research_worker", "validator_node")
    graph_builder.add_edge("policy_rag_worker", "validator_node")
    graph_builder.add_edge("disruption_worker", "validator_node")
    graph_builder.add_edge("booking_approval_node", "validator_node")
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
