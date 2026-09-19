"""
Unit tests for ZICO Tavily web research integration layer.

Covers:
- Test 27: Client initialization with centralized configuration
- Test 28: Successful search with field normalization (title, url, content, score)
- Test 29: Multiple results count, ordering, and independent normalization
- Test 30: Empty results handling (success=True, count=0, no exceptions)
- Test 31: Missing API key raises TavilyConfigError without exposing secrets
- Test 32: Empty API key raises TavilyConfigError
- Test 33: Empty / whitespace-only query validation
- Test 34: Invalid result count validation (0, -1)
- Test 35: Network failure raises TavilyNetworkError
- Test 36: Request timeout raises TavilyTimeoutError
- Test 37: HTTP / provider error handling without secret leakage
- Test 38: Malformed provider response handling (missing results, invalid types)
- Test 39: Missing optional fields handled safely without fabricating values
- Test 40: JSON serialization of response and result models
- Test 41: Secret protection in logs and exception messages
- Test 42: Complete offline test execution with mock transport
"""

import json
from typing import Any, Dict

import httpx
import pytest

from app.core.config import settings
from app.tools.tavily_search import (
    ResearchResult,
    TavilyConfigError,
    TavilyHTTPError,
    TavilyNetworkError,
    TavilyParsingError,
    TavilySearchClient,
    TavilySearchResponse,
    TavilyTimeoutError,
    TavilyValidationError,
)

SAMPLE_TAVILY_PAYLOAD: Dict[str, Any] = {
    "query": "Germany visa requirements for Indian citizens",
    "response_time": 0.45,
    "answer": "Indian citizens require a Schengen visa to travel to Germany.",
    "results": [
        {
            "title": "Germany Visa Requirements - Official Portal",
            "url": "https://www.germany-visa.org/requirements/",
            "content": "Indian passport holders require a valid visa and 6 months passport validity.",
            "score": 0.96,
            "published_date": "2026-01-15",
            "raw_content": "Full parsed page content...",
        },
        {
            "title": "Federal Foreign Office Germany",
            "url": "https://www.auswaertiges-amt.de/en/visa-service",
            "content": "General regulations and Schengen visa fees for tourist travelers.",
            "score": 0.89,
            "published_date": "2026-02-01",
        },
    ],
}


@pytest.fixture(autouse=True)
def reset_tavily_settings(monkeypatch: pytest.MonkeyPatch):
    """Ensure TAVILY_API_KEY is safely configured with a dummy key for testing."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "test-mock-tavily-key")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_client_initialization(monkeypatch: pytest.MonkeyPatch):
    """Test 27: Verify client initializes correctly from centralized settings."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "test-tavily-key-configured")
    client = TavilySearchClient()
    assert client._api_key == "test-tavily-key-configured"


async def test_successful_search():
    """Test 28: Mock realistic successful Tavily response and verify normalization."""
    captured_request: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(200, json=SAMPLE_TAVILY_PAYLOAD)

    async with TavilySearchClient(transport=httpx.MockTransport(handler)) as client:
        response: TavilySearchResponse = await client.search(
            query="Germany visa requirements",
            max_results=2,
        )

    # Verify request payload
    assert captured_request["body"]["api_key"] == "test-mock-tavily-key"
    assert captured_request["body"]["query"] == "Germany visa requirements"
    assert captured_request["body"]["max_results"] == 2

    # Verify response normalization
    assert response.success is True
    assert response.count == 2
    assert response.query == "Germany visa requirements"
    assert response.answer == "Indian citizens require a Schengen visa to travel to Germany."

    first_item = response.results[0]
    assert first_item.title == "Germany Visa Requirements - Official Portal"
    assert first_item.url == "https://www.germany-visa.org/requirements/"
    assert "Indian passport holders" in first_item.content
    assert first_item.score == 0.96
    assert first_item.published_date == "2026-01-15"


async def test_multiple_results_ordering():
    """Test 29: Verify multiple results preserve count, ordering, and independent normalization."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SAMPLE_TAVILY_PAYLOAD)

    async with TavilySearchClient(transport=httpx.MockTransport(handler)) as client:
        response = await client.search(query="Multiple query", max_results=5)

    assert len(response.results) == 2
    assert response.results[0].title == "Germany Visa Requirements - Official Portal"
    assert response.results[1].title == "Federal Foreign Office Germany"
    assert response.results[0].score > response.results[1].score


async def test_empty_results():
    """Test 30: Successful search with zero results returns clean empty collection."""
    empty_payload = {
        "query": "Unmatched query",
        "response_time": 0.12,
        "results": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=empty_payload)

    async with TavilySearchClient(transport=httpx.MockTransport(handler)) as client:
        response = await client.search(query="Unmatched query")

    assert response.success is True
    assert response.count == 0
    assert response.results == []


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test 31: Missing API key raises TavilyConfigError without making network request."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", None)

    with pytest.raises(TavilyConfigError) as exc_info:
        TavilySearchClient(api_key=None)

    assert "missing or empty" in str(exc_info.value)
    assert "test-mock-tavily-key" not in str(exc_info.value)


def test_empty_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test 32: Empty/whitespace API key raises TavilyConfigError."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "   ")

    with pytest.raises(TavilyConfigError) as exc_info:
        TavilySearchClient(api_key="")

    assert "missing or empty" in str(exc_info.value)


async def test_empty_query_validation():
    """Test 33: Empty and whitespace-only query strings fail validation before network call."""
    client = TavilySearchClient(api_key="test-key")

    with pytest.raises(TavilyValidationError) as exc_1:
        await client.search(query="")
    assert "must not be empty" in str(exc_1.value)

    with pytest.raises(TavilyValidationError) as exc_2:
        await client.search(query="    ")
    assert "must not be empty" in str(exc_2.value)


async def test_invalid_max_results_validation():
    """Test 34: Non-positive max_results values fail validation before network call."""
    client = TavilySearchClient(api_key="test-key")

    with pytest.raises(TavilyValidationError) as exc_zero:
        await client.search(query="test", max_results=0)
    assert "positive integer >= 1" in str(exc_zero.value)

    with pytest.raises(TavilyValidationError) as exc_neg:
        await client.search(query="test", max_results=-1)
    assert "positive integer >= 1" in str(exc_neg.value)


async def test_network_failure():
    """Test 35: Network connection failure raises TavilyNetworkError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to connect to Tavily host")

    async with TavilySearchClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TavilyNetworkError) as exc_info:
            await client.search(query="test query")

    assert "Network failure" in str(exc_info.value)


async def test_timeout():
    """Test 36: Request timeout raises TavilyTimeoutError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request read timeout after 10s")

    async with TavilySearchClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TavilyTimeoutError) as exc_info:
            await client.search(query="test query")

    assert "timed out" in str(exc_info.value).lower()


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
async def test_http_error_handling(status_code: int):
    """Test 37: HTTP error status codes raise TavilyHTTPError with status code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"detail": {"error": f"Error code {status_code}"}},
        )

    async with TavilySearchClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TavilyHTTPError) as exc_info:
            await client.search(query="test query")

    assert exc_info.value.status_code == status_code


async def test_malformed_provider_response():
    """Test 38: Malformed response structures raise TavilyParsingError."""

    # Missing 'results' key
    def handler_missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": "test"})

    async with TavilySearchClient(transport=httpx.MockTransport(handler_missing)) as client:
        with pytest.raises(TavilyParsingError) as exc_info:
            await client.search(query="test")
        assert "missing a valid 'results' list" in str(exc_info.value)

    # Non-string URL
    def handler_bad_url(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"title": "Good Title", "url": 12345}]})

    async with TavilySearchClient(transport=httpx.MockTransport(handler_bad_url)) as client:
        with pytest.raises(TavilyParsingError) as exc_info_url:
            await client.search(query="test")
        assert "Invalid URL type" in str(exc_info_url.value)


async def test_optional_fields_handling():
    """Test 39: Missing optional fields (score, published_date, content) default to None safely."""
    payload = {
        "results": [
            {
                "title": "Minimal Result",
                "url": "https://example.com/minimal",
                # score, published_date, content are omitted
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with TavilySearchClient(transport=httpx.MockTransport(handler)) as client:
        response = await client.search(query="minimal test")

    assert response.count == 1
    item = response.results[0]
    assert item.title == "Minimal Result"
    assert item.url == "https://example.com/minimal"
    assert item.content == ""
    assert item.score is None
    assert item.published_date is None


def test_serialization():
    """Test 40: Normalized search results cleanly serialize to JSON dicts and strings."""
    result = ResearchResult(
        title="Sample Research",
        url="https://example.com",
        content="Research text...",
        score=0.91,
    )
    response = TavilySearchResponse(
        success=True,
        query="Sample",
        count=1,
        results=[result],
    )

    dumped = response.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["results"][0]["title"] == "Sample Research"

    json_str = json.dumps(dumped)
    assert isinstance(json_str, str)
    assert "Sample Research" in json_str


async def test_secret_protection(caplog: pytest.LogCaptureFixture):
    """Test 41: Fake API secret is never logged or leaked into exceptions."""
    secret_key = "test-tavily-key-secret-999"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    with caplog.at_level("DEBUG"):
        async with TavilySearchClient(
            api_key=secret_key, transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(TavilyHTTPError) as exc_info:
                await client.search(query="test secret")

    # Verify secret is not in exception message
    assert secret_key not in str(exc_info.value)
    # Verify secret is not in logs
    assert secret_key not in caplog.text
