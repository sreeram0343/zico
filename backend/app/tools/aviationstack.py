"""
AviationStack flight data provider and integration tool for ZICO.

This module provides an asynchronous client for querying the AviationStack REST API,
normalizing raw flight data into typed, LangGraph-compatible Pydantic models.

Architecture:
    ZICO Agents -> AviationStackClient -> AviationStack REST API -> Normalized Result

Security:
    - Centralized configuration via `app.core.config.settings.AVIATIONSTACK_API_KEY`.
    - API keys are never written to source, printed, logged, or exposed in URLs/exceptions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Suppress transport-level request-line logging from httpx to protect query parameter credentials
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Default base endpoint for AviationStack v1 API
DEFAULT_BASE_URL = "https://api.aviationstack.com/v1"
DEFAULT_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AviationStackError(Exception):
    """Base exception for all AviationStack operations."""


class AviationStackConfigError(AviationStackError):
    """Raised when API key or required client configuration is missing or invalid."""


class AviationStackAPIError(AviationStackError):
    """Raised when AviationStack returns a provider-level error in the JSON body."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"AviationStack API error [{code}]: {message}")
        self.code = code
        self.message = message


class AviationStackHTTPError(AviationStackError):
    """Raised when an HTTP error status (4xx/5xx) is received from the endpoint."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"AviationStack HTTP error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class AviationStackTimeoutError(AviationStackError):
    """Raised when an HTTP request to AviationStack times out."""


class AviationStackNetworkError(AviationStackError):
    """Raised on connection or transport-level failures."""


class AviationStackParsingError(AviationStackError):
    """Raised when the response payload is not valid JSON or violates expected schema."""


# ---------------------------------------------------------------------------
# Helper: URL Sanitizer
# ---------------------------------------------------------------------------


def sanitize_url(url: Any) -> str:
    """
    Remove sensitive query parameters (access_key) from URLs for safe logging and error reporting.
    """
    return re.sub(r"access_key=[^&]+", "access_key=[REDACTED]", str(url))


# ---------------------------------------------------------------------------
# Normalized Domain Models (JSON-serializable)
# ---------------------------------------------------------------------------


class AirportDetails(BaseModel):
    """Normalized departure or arrival airport details."""

    airport: Optional[str] = None
    timezone: Optional[str] = None
    iata: Optional[str] = None
    icao: Optional[str] = None
    terminal: Optional[str] = None
    gate: Optional[str] = None
    delay: Optional[int] = None
    scheduled: Optional[str] = None
    estimated: Optional[str] = None
    actual: Optional[str] = None


class AirlineDetails(BaseModel):
    """Normalized operating airline details."""

    name: Optional[str] = None
    iata: Optional[str] = None
    icao: Optional[str] = None


class FlightDetails(BaseModel):
    """Normalized flight designator details."""

    number: Optional[str] = None
    iata: Optional[str] = None
    icao: Optional[str] = None


class NormalizedFlight(BaseModel):
    """
    Structured, normalized representation of a single AviationStack flight record.
    """

    flight_date: Optional[str] = None
    flight_status: Optional[str] = None
    flight: FlightDetails = Field(default_factory=FlightDetails)
    airline: AirlineDetails = Field(default_factory=AirlineDetails)
    departure: AirportDetails = Field(default_factory=AirportDetails)
    arrival: AirportDetails = Field(default_factory=AirportDetails)

    # Convenience flattened fields for straightforward agent consumption
    flight_number: Optional[str] = None
    flight_iata: Optional[str] = None
    flight_icao: Optional[str] = None
    airline_name: Optional[str] = None
    airline_iata: Optional[str] = None
    airline_icao: Optional[str] = None
    departure_airport: Optional[str] = None
    departure_iata: Optional[str] = None
    departure_scheduled: Optional[str] = None
    departure_estimated: Optional[str] = None
    departure_actual: Optional[str] = None
    arrival_airport: Optional[str] = None
    arrival_iata: Optional[str] = None
    arrival_scheduled: Optional[str] = None
    arrival_estimated: Optional[str] = None
    arrival_actual: Optional[str] = None


class PaginationInfo(BaseModel):
    """AviationStack pagination metadata."""

    limit: int = 0
    offset: int = 0
    count: int = 0
    total: int = 0


class AviationStackResponse(BaseModel):
    """
    Standard envelope returned by AviationStackClient methods.
    """

    success: bool = True
    count: int = 0
    pagination: Optional[PaginationInfo] = None
    data: List[NormalizedFlight] = Field(default_factory=list)
    provider: str = "aviationstack"


# ---------------------------------------------------------------------------
# Normalization Helper
# ---------------------------------------------------------------------------


def normalize_flight_record(raw: Dict[str, Any]) -> NormalizedFlight:
    """
    Transforms a raw AviationStack flight JSON object into a normalized, typed model.
    """
    dep = raw.get("departure") or {}
    arr = raw.get("arrival") or {}
    airline = raw.get("airline") or {}
    flight = raw.get("flight") or {}

    flight_obj = FlightDetails(
        number=flight.get("number"),
        iata=flight.get("iata"),
        icao=flight.get("icao"),
    )
    airline_obj = AirlineDetails(
        name=airline.get("name"),
        iata=airline.get("iata"),
        icao=airline.get("icao"),
    )
    dep_obj = AirportDetails(
        airport=dep.get("airport"),
        timezone=dep.get("timezone"),
        iata=dep.get("iata"),
        icao=dep.get("icao"),
        terminal=dep.get("terminal"),
        gate=dep.get("gate"),
        delay=dep.get("delay"),
        scheduled=dep.get("scheduled"),
        estimated=dep.get("estimated"),
        actual=dep.get("actual"),
    )
    arr_obj = AirportDetails(
        airport=arr.get("airport"),
        timezone=arr.get("timezone"),
        iata=arr.get("iata"),
        icao=arr.get("icao"),
        terminal=arr.get("terminal"),
        gate=arr.get("gate"),
        delay=arr.get("delay"),
        scheduled=arr.get("scheduled"),
        estimated=arr.get("estimated"),
        actual=arr.get("actual"),
    )

    return NormalizedFlight(
        flight_date=raw.get("flight_date"),
        flight_status=raw.get("flight_status"),
        flight=flight_obj,
        airline=airline_obj,
        departure=dep_obj,
        arrival=arr_obj,
        flight_number=flight_obj.number,
        flight_iata=flight_obj.iata,
        flight_icao=flight_obj.icao,
        airline_name=airline_obj.name,
        airline_iata=airline_obj.iata,
        airline_icao=airline_obj.icao,
        departure_airport=dep_obj.airport,
        departure_iata=dep_obj.iata,
        departure_scheduled=dep_obj.scheduled,
        departure_estimated=dep_obj.estimated,
        departure_actual=dep_obj.actual,
        arrival_airport=arr_obj.airport,
        arrival_iata=arr_obj.iata,
        arrival_scheduled=arr_obj.scheduled,
        arrival_estimated=arr_obj.estimated,
        arrival_actual=arr_obj.actual,
    )


# ---------------------------------------------------------------------------
# Asynchronous AviationStack Client
# ---------------------------------------------------------------------------


class AviationStackClient:
    """
    Asynchronous client for querying AviationStack flight tracking and status endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.AsyncClient] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        """
        Initialize the AviationStack client.

        Args:
            api_key: Optional API key override. Defaults to settings.AVIATIONSTACK_API_KEY.
            base_url: Optional base endpoint URL override. Defaults to DEFAULT_BASE_URL.
            timeout: Request timeout in seconds.
            client: Optional pre-configured httpx.AsyncClient instance.
            transport: Optional custom transport (e.g. httpx.MockTransport for tests).

        Raises:
            AviationStackConfigError: If the API key is absent or empty.
        """
        resolved_key = (
            api_key if api_key is not None else getattr(settings, "AVIATIONSTACK_API_KEY", "")
        )
        if not resolved_key or not isinstance(resolved_key, str) or not resolved_key.strip():
            raise AviationStackConfigError(
                "AviationStack API key is missing or empty. Please configure AVIATIONSTACK_API_KEY."
            )

        self._api_key = resolved_key.strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

        if client is not None:
            self._client = client
            self._manage_client = False
        elif transport is not None:
            self._client = httpx.AsyncClient(transport=transport, timeout=self.timeout)
            self._manage_client = True
        else:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._manage_client = True

    async def __aenter__(self) -> "AviationStackClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client session if internally managed."""
        if self._manage_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def get_flights(
        self,
        flight_number: Optional[str] = None,
        flight_iata: Optional[str] = None,
        flight_icao: Optional[str] = None,
        airline_name: Optional[str] = None,
        airline_iata: Optional[str] = None,
        airline_icao: Optional[str] = None,
        dep_iata: Optional[str] = None,
        arr_iata: Optional[str] = None,
        dep_icao: Optional[str] = None,
        arr_icao: Optional[str] = None,
        flight_date: Optional[str] = None,
        flight_status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> AviationStackResponse:
        """
        Query the AviationStack `/flights` endpoint with optional filter parameters.

        Only non-empty, non-None parameters are sent in the HTTP request.

        Returns:
            AviationStackResponse containing normalized flight records and pagination data.

        Raises:
            AviationStackAPIError: When the provider returns an error payload.
            AviationStackHTTPError: When an HTTP 4xx/5xx status is received.
            AviationStackTimeoutError: When the request times out.
            AviationStackNetworkError: On transport or network connectivity errors.
            AviationStackParsingError: When response JSON cannot be decoded.
        """
        endpoint = f"{self.base_url}/flights"

        # Build clean query parameters; omit any None or empty values
        params: Dict[str, Any] = {"access_key": self._api_key}

        raw_filters = {
            "flight_number": flight_number,
            "flight_iata": flight_iata,
            "flight_icao": flight_icao,
            "airline_name": airline_name,
            "airline_iata": airline_iata,
            "airline_icao": airline_icao,
            "dep_iata": dep_iata,
            "arr_iata": arr_iata,
            "dep_icao": dep_icao,
            "arr_icao": arr_icao,
            "flight_date": flight_date,
            "flight_status": flight_status,
            "limit": limit,
            "offset": offset,
        }

        for k, v in raw_filters.items():
            if v is not None:
                if isinstance(v, str):
                    cleaned = v.strip()
                    if cleaned:
                        params[k] = cleaned
                elif isinstance(v, (int, float)):
                    params[k] = v

        # Operational logging (never log access_key)
        safe_param_keys = [k for k in params if k != "access_key"]
        logger.info("Executing AviationStack flight lookup (filters=%s)", safe_param_keys)

        try:
            response = await self._client.get(endpoint, params=params)
        except httpx.TimeoutException as exc:
            logger.warning("AviationStack request timed out after %ss", self.timeout)
            raise AviationStackTimeoutError(
                f"AviationStack request timed out after {self.timeout}s"
            ) from exc
        except httpx.RequestError as exc:
            safe_msg = sanitize_url(str(exc))
            logger.error("AviationStack network failure: %s", safe_msg)
            raise AviationStackNetworkError(
                f"Network failure communicating with AviationStack: {safe_msg}"
            ) from exc

        # Check HTTP status
        if response.status_code >= 400:
            logger.error("AviationStack returned HTTP error status %d", response.status_code)
            raise AviationStackHTTPError(
                status_code=response.status_code,
                message=f"HTTP status {response.status_code} received from provider",
            )

        # Parse JSON
        try:
            payload = response.json()
        except Exception as exc:
            logger.error("Failed to parse AviationStack JSON response: %s", exc)
            raise AviationStackParsingError(
                f"AviationStack response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise AviationStackParsingError("AviationStack response payload must be a JSON object")

        # Check for provider API errors embedded in 200 responses
        if "error" in payload and payload["error"]:
            err = payload["error"]
            if isinstance(err, dict):
                code = str(err.get("code", "unknown"))
                msg = str(err.get("message", "Unknown AviationStack error"))
            else:
                code = "unknown"
                msg = str(err)
            logger.warning("AviationStack API returned error [%s]: %s", code, msg)
            raise AviationStackAPIError(code=code, message=msg)

        # Normalize flight records
        raw_data = payload.get("data")
        if not isinstance(raw_data, list):
            raw_data = []

        normalized_flights = [
            normalize_flight_record(item) for item in raw_data if isinstance(item, dict)
        ]

        # Extract pagination if provided
        pagination: Optional[PaginationInfo] = None
        if "pagination" in payload and isinstance(payload["pagination"], dict):
            p = payload["pagination"]
            pagination = PaginationInfo(
                limit=p.get("limit", len(normalized_flights)),
                offset=p.get("offset", 0),
                count=p.get("count", len(normalized_flights)),
                total=p.get("total", len(normalized_flights)),
            )

        if not normalized_flights:
            logger.info("AviationStack returned 0 matching flight records")
        else:
            logger.info(
                "AviationStack flight lookup completed successfully (results=%d)",
                len(normalized_flights),
            )

        return AviationStackResponse(
            success=True,
            count=len(normalized_flights),
            pagination=pagination,
            data=normalized_flights,
        )

    # Method alias for intuitive agent calls
    search_flights = get_flights
