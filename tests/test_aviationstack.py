"""
Unit tests for ZICO AviationStack flight data integration tool.

Covers:
- Test 26: Successful flight response parsing and normalization
- Test 27: Flight number query parameter construction
- Test 28: Airport-based query parameter construction
- Test 29: Optional parameter omission (no None or empty query strings)
- Test 30: Empty result handling (successful request with 0 matches)
- Test 31: Provider-level API error detection and parsing
- Test 32: HTTP error status handling (401, 403, 429, 500)
- Test 33: Request timeout handling
- Test 34: Network/connection failure handling
- Test 35: Malformed JSON response parsing
- Test 36: Missing API key raises AviationStackConfigError
- Test 37: Empty API key raises AviationStackConfigError
- Test 38: Secret protection (credentials never logged or leaked in exceptions)
- Test 39: Async client context manager lifecycle cleanup
- Test 40: JSON serialization of normalized response models
"""

import json
from typing import Any, Dict
import httpx
import pytest

from app.core.config import settings
from app.tools.aviationstack import (
    AviationStackAPIError,
    AviationStackClient,
    AviationStackConfigError,
    AviationStackHTTPError,
    AviationStackNetworkError,
    AviationStackParsingError,
    AviationStackResponse,
    AviationStackTimeoutError,
)

SAMPLE_FLIGHT_DATA: Dict[str, Any] = {
    "pagination": {
        "limit": 10,
        "offset": 0,
        "count": 1,
        "total": 1,
    },
    "data": [
        {
            "flight_date": "2026-09-16",
            "flight_status": "scheduled",
            "departure": {
                "airport": "Thiruvananthapuram International",
                "timezone": "Asia/Kolkata",
                "iata": "TRV",
                "icao": "VOTV",
                "terminal": "2",
                "gate": "B3",
                "delay": 15,
                "scheduled": "2026-09-16T10:30:00+00:00",
                "estimated": "2026-09-16T10:45:00+00:00",
                "actual": None,
            },
            "arrival": {
                "airport": "Dubai International",
                "timezone": "Asia/Dubai",
                "iata": "DXB",
                "icao": "OMDB",
                "terminal": "3",
                "gate": "A12",
                "delay": None,
                "scheduled": "2026-09-16T13:15:00+00:00",
                "estimated": "2026-09-16T13:15:00+00:00",
                "actual": None,
            },
            "airline": {
                "name": "Emirates",
                "iata": "EK",
                "icao": "UAE",
            },
            "flight": {
                "number": "522",
                "iata": "EK522",
                "icao": "UAE522",
            },
        }
    ],
}


@pytest.fixture(autouse=True)
def reset_aviation_settings(monkeypatch: pytest.MonkeyPatch):
    """Ensure AVIATIONSTACK_API_KEY is safely configured with a dummy key for testing."""
    monkeypatch.setattr(settings, "AVIATIONSTACK_API_KEY", "test-mock-aviation-key")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


async def test_successful_flight_response():
    """Test 26: Mock realistic successful AviationStack response and verify normalization."""
    captured_request: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["params"] = dict(request.url.params)
        return httpx.Response(200, json=SAMPLE_FLIGHT_DATA)

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        response: AviationStackResponse = await client.get_flights(dep_iata="TRV", arr_iata="DXB")

    assert "/flights" in captured_request["url"]
    assert captured_request["params"]["access_key"] == "test-mock-aviation-key"
    assert captured_request["params"]["dep_iata"] == "TRV"
    assert captured_request["params"]["arr_iata"] == "DXB"

    assert response.success is True
    assert response.count == 1
    assert len(response.data) == 1

    flight = response.data[0]
    assert flight.flight_iata == "EK522"
    assert flight.flight_number == "522"
    assert flight.airline_name == "Emirates"
    assert flight.departure_iata == "TRV"
    assert flight.departure_airport == "Thiruvananthapuram International"
    assert flight.arrival_iata == "DXB"
    assert flight.arrival_airport == "Dubai International"
    assert flight.flight_status == "scheduled"
    assert flight.departure_scheduled == "2026-09-16T10:30:00+00:00"


async def test_flight_number_lookup():
    """Test 27: Query with flight number and verify correct provider parameter construction."""
    captured_params: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(dict(request.url.params))
        return httpx.Response(200, json=SAMPLE_FLIGHT_DATA)

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        await client.get_flights(flight_iata="EK522", flight_number="522")

    assert captured_params["flight_iata"] == "EK522"
    assert captured_params["flight_number"] == "522"


async def test_airport_based_search():
    """Test 28: Search by departure and arrival airport codes."""
    captured_params: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(dict(request.url.params))
        return httpx.Response(200, json=SAMPLE_FLIGHT_DATA)

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        await client.get_flights(dep_iata="TRV", arr_iata="DXB")

    assert captured_params["dep_iata"] == "TRV"
    assert captured_params["arr_iata"] == "DXB"


async def test_optional_parameter_omission():
    """Test 29: Only supplied non-empty parameters are included in the request."""
    captured_params: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(dict(request.url.params))
        return httpx.Response(200, json=SAMPLE_FLIGHT_DATA)

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        await client.get_flights(flight_iata="AI123", arr_iata=None, airline_name="   ")

    assert "flight_iata" in captured_params
    assert "arr_iata" not in captured_params
    assert "airline_name" not in captured_params
    # Ensure no literal "None" query parameters exist
    assert "None" not in captured_params.values()


async def test_empty_result():
    """Test 30: Valid response containing zero matching records is not treated as an error."""
    empty_payload = {
        "pagination": {"limit": 10, "offset": 0, "count": 0, "total": 0},
        "data": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=empty_payload)

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        response = await client.get_flights(dep_iata="XYZ", arr_iata="ABC")

    assert response.success is True
    assert response.count == 0
    assert response.data == []


async def test_provider_api_error():
    """Test 31: Provider error embedded in JSON response raises AviationStackAPIError."""
    error_payload = {
        "error": {
            "code": "usage_limit_reached",
            "message": "Your monthly usage limit has been reached.",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=error_payload)

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AviationStackAPIError) as exc_info:
            await client.get_flights(dep_iata="TRV")

    assert exc_info.value.code == "usage_limit_reached"
    assert "Your monthly usage limit has been reached." in str(exc_info.value)


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
async def test_http_errors(status_code: int):
    """Test 32: HTTP error codes raise AviationStackHTTPError with status code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="HTTP Error Encountered")

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AviationStackHTTPError) as exc_info:
            await client.get_flights()

    assert exc_info.value.status_code == status_code


async def test_timeout():
    """Test 33: Request timeout raises AviationStackTimeoutError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out after 10s")

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AviationStackTimeoutError) as exc_info:
            await client.get_flights()

    assert "timed out" in str(exc_info.value).lower()


async def test_network_failure():
    """Test 34: Network failure raises AviationStackNetworkError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to establish a new connection: DNS lookup failure")

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AviationStackNetworkError) as exc_info:
            await client.get_flights()

    assert "Network failure" in str(exc_info.value)


async def test_malformed_json():
    """Test 35: Non-JSON response body raises AviationStackParsingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>502 Bad Gateway</body></html>")

    async with AviationStackClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AviationStackParsingError) as exc_info:
            await client.get_flights()

    assert "not valid JSON" in str(exc_info.value)


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test 36: Missing API key raises AviationStackConfigError without network call."""
    monkeypatch.setattr(settings, "AVIATIONSTACK_API_KEY", None)

    with pytest.raises(AviationStackConfigError) as exc_info:
        AviationStackClient(api_key=None)

    assert "missing or empty" in str(exc_info.value)


def test_empty_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test 37: Empty/whitespace API key raises AviationStackConfigError."""
    monkeypatch.setattr(settings, "AVIATIONSTACK_API_KEY", "   ")

    with pytest.raises(AviationStackConfigError) as exc_info:
        AviationStackClient(api_key="")

    assert "missing or empty" in str(exc_info.value)


async def test_secret_protection(caplog: pytest.LogCaptureFixture):
    """Test 38: Fake API secret is never leaked into logs, exceptions, or error representations."""
    secret_key = "test-aviationstack-secret-sensitive-999"

    def handler(request: httpx.Request) -> httpx.Response:
        # Simulate an error response
        return httpx.Response(500, text="Internal Server Error")

    with caplog.at_level("DEBUG"):
        async with AviationStackClient(api_key=secret_key, transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(AviationStackHTTPError) as exc_info:
                await client.get_flights()

    # Verify secret is not in the raised exception string
    assert secret_key not in str(exc_info.value)

    # Verify secret is not in captured logs
    assert secret_key not in caplog.text


async def test_client_cleanup():
    """Test 39: Async context manager properly closes internal HTTP client."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = AviationStackClient(api_key="test-key", transport=httpx.MockTransport(handler))
    assert client._client.is_closed is False

    async with client:
        await client.get_flights()

    assert client._client.is_closed is True


def test_serialization():
    """Test 40: Normalized AviationStackResponse cleanly serializes to JSON."""
    raw_flight = SAMPLE_FLIGHT_DATA["data"][0]
    from app.tools.aviationstack import normalize_flight_record

    flight = normalize_flight_record(raw_flight)
    response = AviationStackResponse(
        success=True,
        count=1,
        data=[flight],
    )

    dumped = response.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["data"][0]["flight_iata"] == "EK522"

    json_str = json.dumps(dumped)
    assert isinstance(json_str, str)
    assert "EK522" in json_str
