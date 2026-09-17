"""
Unit and integration test suite for the ZICO FastAPI application entry point.

Tests cover:
    - Application creation and ASGI instance validity
    - Application metadata (title, description, version)
    - Lightweight liveness endpoint (GET /health)
    - Health endpoint isolation (zero external provider or workflow calls)
    - Route registration (/api/v1/chat mounted properly)
    - Duplicate route prevention
    - Supported / unsupported HTTP methods on /health
    - Chat route execution via mocked workflow
    - Input validation on chat endpoint (empty message -> 422)
    - OpenAPI schema generation and docs (/openapi.json, /docs)
    - Import safety (no external calls, no blocking server startup)
    - Logging initialization and idempotency
    - Source code hygiene (no business logic, no secrets)
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router as chat_router
from app.api.schemas import ResponseStatus, TravelResponse
from app.core.logging import configure_logging
from app.main import app, health_check


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient bound to the ZICO FastAPI application."""
    return TestClient(app)


def _collect_all_routes(application: FastAPI) -> List[Tuple[str, Set[str]]]:
    """
    Extract (path, methods) pairs from application routes including included routers.
    """
    routes: List[Tuple[str, Set[str]]] = []
    for r in application.routes:
        if hasattr(r, "path") and hasattr(r, "methods"):
            routes.append((r.path, set(r.methods or [])))
        elif hasattr(r, "original_router"):
            for sub_r in r.original_router.routes:
                if hasattr(sub_r, "path") and hasattr(sub_r, "methods"):
                    routes.append((sub_r.path, set(sub_r.methods or [])))
    return routes


# ===========================================================================
# 1. Application Creation & Metadata
# ===========================================================================

def test_app_is_fastapi_instance() -> None:
    """Verify that app is an authentic FastAPI application instance."""
    assert isinstance(app, FastAPI)


def test_app_metadata() -> None:
    """Verify application title, version, and concise description."""
    assert app.title == "ZICO"
    assert app.version == "0.1.0"
    assert "Travel Operations Assistant" in app.description


def test_no_unimplemented_claims_in_description() -> None:
    """Verify description does not claim unbuilt features like booking or payments."""
    desc = app.description.lower()
    for forbidden in ["payment", "booking", "autonomous booking", "real-time guarantee"]:
        assert forbidden not in desc


# ===========================================================================
# 2. Health Endpoint
# ===========================================================================

def test_health_endpoint_success(client: TestClient) -> None:
    """Verify GET /health returns 200 OK and expected JSON body."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_check_function_direct() -> None:
    """Verify the health_check endpoint function can be called directly."""
    result = await health_check()
    assert result == {"status": "ok"}


def test_health_endpoint_methods(client: TestClient) -> None:
    """Verify /health accepts GET and rejects other HTTP methods with 405."""
    assert client.get("/health").status_code == 200
    assert client.post("/health").status_code == 405
    assert client.put("/health").status_code == 405
    assert client.delete("/health").status_code == 405


def test_health_endpoint_isolation(client: TestClient) -> None:
    """
    Verify GET /health performs zero external calls or workflow executions.
    """
    with patch("app.graph.workflow.run_workflow") as mock_wf, \
         patch("httpx.AsyncClient.request") as mock_httpx, \
         patch("openai.OpenAI") as mock_openai:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        mock_wf.assert_not_called()
        mock_httpx.assert_not_called()
        mock_openai.assert_not_called()


# ===========================================================================
# 3. Route Mounting & Router Verification
# ===========================================================================

def test_chat_route_is_registered() -> None:
    """Verify POST /api/v1/chat is properly mounted on the application."""
    all_routes = _collect_all_routes(app)
    chat_routes = [
        (path, methods)
        for path, methods in all_routes
        if path == "/api/v1/chat" and "POST" in methods
    ]
    assert len(chat_routes) == 1, f"Expected exactly 1 POST /api/v1/chat route, found {len(chat_routes)}"


def test_no_duplicate_chat_route() -> None:
    """Verify /api/v1/chat is not registered multiple times."""
    all_routes = _collect_all_routes(app)
    paths = [path for path, _ in all_routes if path == "/api/v1/chat"]
    assert len(paths) == 1


def test_no_double_prefixed_route() -> None:
    """Ensure route is not inadvertently prefixed twice as /api/v1/api/v1/chat."""
    all_routes = _collect_all_routes(app)
    for path, _ in all_routes:
        assert not path.startswith("/api/v1/api/v1")


def test_no_unwanted_root_endpoint(client: TestClient) -> None:
    """Ensure no root GET / endpoint was added unnecessarily."""
    response = client.get("/")
    assert response.status_code == 404


def test_router_identity() -> None:
    """Verify that the mounted router corresponds to app.api.routes.router."""
    included_routers = [
        r for r in app.routes if getattr(r, "original_router", None) is not None
    ]
    matching = [
        r for r in included_routers if r.original_router is chat_router
    ]
    assert len(matching) == 1


# ===========================================================================
# 4. Chat Endpoint Integration through Application
# ===========================================================================

def test_chat_endpoint_valid_request(client: TestClient) -> None:
    """
    Verify a valid travel request to POST /api/v1/chat reaches the route
    and returns a structured TravelResponse when workflow is mocked.
    """
    mock_final_state: Dict[str, Any] = {
        "user_query": "Find flights from TRV to DXB.",
        "session_id": "test_sess_001",
        "request_id": "req_001",
        "intent": "flight_search",
        "flight_status": "success",
        "research_status": "not_started",
        "validation_status": "passed",
        "workflow_status": "completed",
        "final_response": "Found direct flight AI-967 from TRV to DXB departing at 14:00.",
        "sources": [],
        "errors": [],
        "validation_errors": [],
        "iteration_count": 1,
    }

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state
        payload = {"message": "Find flights from TRV to DXB.", "session_id": "test_sess_001"}
        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Found direct flight AI-967 from TRV to DXB departing at 14:00."
        assert data["session_id"] == "test_sess_001"
        assert data["status"] == "success"
        assert data["sources"] == []
        assert data["errors"] == []

        # Validate with Pydantic model
        validated = TravelResponse.model_validate(data)
        assert validated.status == ResponseStatus.SUCCESS


def test_chat_endpoint_validation_error(client: TestClient) -> None:
    """
    Verify that an empty message triggers standard 422 validation error
    and does not execute the LangGraph workflow.
    """
    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        response = client.post("/api/v1/chat", json={"message": ""})
        assert response.status_code == 422
        mock_run.assert_not_called()


# ===========================================================================
# 5. OpenAPI & Documentation
# ===========================================================================

def test_openapi_schema(client: TestClient) -> None:
    """Verify GET /openapi.json contains expected endpoints and metadata."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    assert schema["info"]["title"] == "ZICO"
    assert schema["info"]["version"] == "0.1.0"
    paths = schema.get("paths", {})
    assert "/health" in paths
    assert "/api/v1/chat" in paths
    assert "get" in paths["/health"]
    assert "post" in paths["/api/v1/chat"]


def test_docs_endpoint(client: TestClient) -> None:
    """Verify GET /docs returns 200 OK and serves Swagger UI HTML."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "html" in response.headers.get("content-type", "")


# ===========================================================================
# 6. Import Safety & Server Behavior
# ===========================================================================

def test_app_import_is_non_blocking() -> None:
    """
    Ensure importing app does not start an active uvicorn server or block execution.
    """
    import app.main as main_mod
    assert hasattr(main_mod, "app")
    assert not getattr(main_mod.app, "_is_running", False)


def test_app_import_no_external_network_activity() -> None:
    """
    Verify reloading app.main does not invoke external network calls.
    """
    with patch("httpx.AsyncClient.request") as mock_httpx_async, \
         patch("httpx.Client.request") as mock_httpx_sync, \
         patch("requests.request") as mock_requests:
        import app.main as main_mod
        importlib.reload(main_mod)

        mock_httpx_async.assert_not_called()
        mock_httpx_sync.assert_not_called()
        mock_requests.assert_not_called()


# ===========================================================================
# 7. Logging Configuration & Idempotency
# ===========================================================================

def test_logging_is_configured_and_idempotent() -> None:
    """
    Verify configure_logging is called and safe against duplicate handlers.
    """
    import logging
    root_logger = logging.getLogger()
    initial_count = len(root_logger.handlers)

    # Call configure_logging again
    configure_logging()
    assert len(root_logger.handlers) == initial_count


# ===========================================================================
# 8. Source Code Architectural Guardrails
# ===========================================================================

def test_main_source_has_no_secret_names() -> None:
    """
    Ensure app/main.py contains no hardcoded secrets or environment reading.
    """
    import app.main as main_mod
    main_path = Path(main_mod.__file__).resolve()
    source_code = main_path.read_text(encoding="utf-8")

    forbidden_tokens = [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "AVIATIONSTACK_API_KEY",
        "os.getenv",
        "os.environ",
    ]
    for token in forbidden_tokens:
        assert token not in source_code, f"Forbidden token '{token}' found in app/main.py"


def test_main_source_has_no_business_logic_or_external_providers() -> None:
    """
    Ensure app/main.py contains no direct LLM providers, database imports, or workflow graph construction.
    """
    import app.main as main_mod
    main_path = Path(main_mod.__file__).resolve()
    source_code = main_path.read_text(encoding="utf-8")

    forbidden_tokens = [
        "ChatOpenAI",
        "OpenAI(",
        "TavilyClient",
        "AviationStack",
        "StateGraph",
        "ainvoke",
        "invoke",
        "CORSMiddleware",
        "redis",
        "asyncpg",
        "sqlalchemy",
    ]
    for token in forbidden_tokens:
        assert token not in source_code, f"Forbidden token '{token}' found in app/main.py"
