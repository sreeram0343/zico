from datetime import datetime, timedelta
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph.engine import create_zico_graph
from app.graph.state import (
    ActionStatus,
    ActionType,
    Location,
    PendingAction,
    SegmentType,
    TripSegment,
)


def _make_test_segment() -> TripSegment:
    now = datetime(2026, 9, 15, 10, 0)
    return TripSegment(
        id="seg_flight_101",
        type=SegmentType.FLIGHT,
        title="Flight BA 101: JFK -> LHR",
        start_time=now,
        end_time=now + timedelta(hours=7),
        location=Location(name="London Heathrow", iata_code="LHR"),
        cost=550.0,
        is_confirmed=False,
    )


def test_hitl_interrupt_approval_workflow():
    """Verify graph pauses at booking_approval_node on high-impact action and resumes on approval."""
    checkpointer = MemorySaver()
    graph = create_zico_graph(checkpointer=checkpointer)

    seg = _make_test_segment()
    action = PendingAction(
        action_id="act_confirm_ba101",
        action_type=ActionType.BOOKING,
        description="Book Flight BA 101 for $550 USD",
        payload={"segment_id": seg.id, "cost": 550.0},
        requires_explicit_approval=True,
        status=ActionStatus.PENDING,
    )

    thread_config = {"configurable": {"thread_id": "thread_hitl_approval_test"}}

    initial_state = {
        "trip_id": "trip_test_101",
        "user_id": "user_traveler",
        "itinerary": [seg],
        "pending_actions": [action],
        "messages": [],
        "next_node": "booking_approval_node",
    }

    # 1. First execution should hit interrupt() at booking_approval_node
    stream_events = list(graph.stream(initial_state, thread_config))
    assert len(stream_events) > 0

    # Verify graph state is paused and waiting for human approval
    current_state = graph.get_state(thread_config)
    assert len(current_state.tasks) > 0
    assert len(current_state.tasks[0].interrupts) > 0
    interrupt_data = current_state.tasks[0].interrupts[0].value
    assert interrupt_data["action_id"] == "act_confirm_ba101"
    assert interrupt_data["action_type"] == "BOOKING"

    # 2. Resume execution with human approval Command
    resume_command = Command(resume={"approved": True, "actor": "lead_traveler"})
    final_events = list(graph.stream(resume_command, thread_config))
    assert len(final_events) > 0

    final_state = graph.get_state(thread_config)
    assert len(final_state.tasks) == 0  # Completed execution
    saved_actions = final_state.values.get("pending_actions", [])
    assert len(saved_actions) == 1
    assert saved_actions[0].status == ActionStatus.APPROVED

    saved_itinerary = final_state.values.get("itinerary", [])
    assert len(saved_itinerary) == 1
    assert saved_itinerary[0].is_confirmed is True


def test_hitl_interrupt_rejection_workflow():
    """Verify graph marks action as REJECTED when traveler denies approval."""
    checkpointer = MemorySaver()
    graph = create_zico_graph(checkpointer=checkpointer)

    seg = _make_test_segment()
    action = PendingAction(
        action_id="act_cancel_hotel",
        action_type=ActionType.CANCELLATION,
        description="Cancel non-refundable hotel booking",
        payload={"segment_id": seg.id},
        requires_explicit_approval=True,
        status=ActionStatus.PENDING,
    )

    thread_config = {"configurable": {"thread_id": "thread_hitl_rejection_test"}}

    initial_state = {
        "trip_id": "trip_test_102",
        "user_id": "user_traveler",
        "itinerary": [seg],
        "pending_actions": [action],
        "messages": [],
        "next_node": "booking_approval_node",
    }

    # 1. Run until interrupt
    list(graph.stream(initial_state, thread_config))

    # 2. Resume with rejection Command
    resume_command = Command(resume={"approved": False})
    list(graph.stream(resume_command, thread_config))

    final_state = graph.get_state(thread_config)
    saved_actions = final_state.values.get("pending_actions", [])
    assert len(saved_actions) == 1
    assert saved_actions[0].status == ActionStatus.REJECTED
    # Itinerary remains unconfirmed
    assert final_state.values["itinerary"][0].is_confirmed is False
