"""
Concurrency and State Isolation Test Suite for ZICO.

Validates that concurrent requests executing simultaneously maintain complete
isolation across:
    - Request context variables (X-Request-ID, session_id)
    - TravelState dictionaries and graph execution copies
    - Domain agent execution states
    - Tool results and error dictionaries
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.state import create_initial_state
from app.graph.workflow import run_workflow
from app.main import app


@pytest.mark.asyncio
async def test_concurrent_request_context_isolation():
    """Verify concurrent requests do NOT leak request_id or session_id across async contexts."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        async def make_request(req_num: int):
            custom_req_id = f"req_iso_{req_num}_{asyncio.current_task().get_name()}"
            custom_sess_id = f"sess_iso_{req_num}"
            headers = {
                "X-Request-ID": custom_req_id,
                "X-Session-ID": custom_sess_id,
            }
            payload = {
                "message": f"Travel request number {req_num}",
                "session_id": custom_sess_id,
            }

            resp = await client.post("/api/v1/chat", json=payload, headers=headers)
            assert resp.status_code in [200, 400, 500, 503]
            resp_req_id = resp.headers.get("X-Request-ID")
            assert resp_req_id == custom_req_id
            return resp_req_id, resp.json().get("session_id")

        # Run 10 concurrent requests simultaneously
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All request IDs and session IDs must be unique
        returned_req_ids = [r[0] for r in results]
        returned_sess_ids = [r[1] for r in results]

        assert len(set(returned_req_ids)) == 10
        assert len(set(returned_sess_ids)) == 10


@pytest.mark.asyncio
async def test_travel_state_deep_copy_isolation():
    """Verify distinct TravelState initializations are mutually isolated."""
    state1 = create_initial_state("Query 1", session_id="s1", request_id="r1")
    state2 = create_initial_state("Query 2", session_id="s2", request_id="r2")

    state1["flight_results"] = [{"flight_iata": "AI101"}]
    state1["errors"].append("Error in 1")

    assert len(state2["flight_results"]) == 0
    assert len(state2["errors"]) == 0
    assert state1["user_query"] != state2["user_query"]
    assert state1["session_id"] != state2["session_id"]


@pytest.mark.asyncio
async def test_concurrent_workflow_execution_isolation():
    """Verify run_workflow maintains strict state isolation under simultaneous invocations."""
    state_a = create_initial_state("Plan trip to Dubai", session_id="sess_A", request_id="req_A")
    state_b = create_initial_state("Find flight to London", session_id="sess_B", request_id="req_B")

    async def run_a():
        return await run_workflow(state_a)

    async def run_b():
        return await run_workflow(state_b)

    res_a, res_b = await asyncio.gather(run_a(), run_b())

    assert res_a["session_id"] == "sess_A"
    assert res_b["session_id"] == "sess_B"
    assert res_a["request_id"] == "req_A"
    assert res_b["request_id"] == "req_B"
    assert res_a["user_query"] != res_b["user_query"]
