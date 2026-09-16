"""
Tavily web research integration layer for ZICO.

This module provides an asynchronous client for performing targeted web research
queries via the Tavily REST API, normalizing search results into typed,
LangGraph-compatible Pydantic models.

Architecture:
    ZICO Research Agent -> TavilySearchClient -> Tavily REST API -> Normalized Results

Security:
    - Centralized configuration via `app.core.config.settings.TAVILY_API_KEY`.
    - API keys are never printed, logged, or exposed in exceptions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Default base endpoint and timeout for Tavily API
DEFAULT_TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TavilySearchError(Exception):
    """Base exception for all Tavily search operations."""


class TavilyConfigError(TavilySearchError):
    """Raised when the Tavily API key or configuration is missing or invalid."""


class TavilyValidationError(TavilySearchError, ValueError):
    """Raised when search query parameters fail validation before execution."""


class TavilyAPIError(TavilySearchError):
    """Raised when Tavily returns an error response payload."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(f"Tavily API error: {message}")
        self.message = message
        self.status_code = status_code


class TavilyHTTPError(TavilySearchError):
    """Raised when an HTTP error status (4xx/5xx) is received from Tavily."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Tavily HTTP error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class TavilyTimeoutError(TavilySearchError):
    """Raised when an HTTP request to Tavily times out."""


class TavilyNetworkError(TavilySearchError):
    """Raised on transport or network connectivity failures."""


class TavilyParsingError(TavilySearchError):
    """Raised when the response cannot be decoded as valid JSON or violates expected schema."""


# ---------------------------------------------------------------------------
# Domain Models (LangGraph / JSON-serializable)
# ---------------------------------------------------------------------------


class ResearchResult(BaseModel):
    """
    Normalized research finding preserving source provenance and metadata.
    """

    title: str
    url: str
    content: str
    score: Optional[float] = None
    published_date: Optional[str] = None
    raw_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TavilySearchResponse(BaseModel):
    """
    Envelope returned by TavilySearchClient search operations.
    """

    success: bool = True
    query: str
    count: int = 0
    results: List[ResearchResult] = Field(default_factory=list)
    answer: Optional[str] = None
    response_time: Optional[float] = None
    provider: str = "tavily"


# ---------------------------------------------------------------------------
# Tavily Search Client
# ---------------------------------------------------------------------------


class TavilySearchClient:
    """
    Asynchronous client for executing web research queries against Tavily.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.AsyncClient] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        """
        Initialize the Tavily search client.

        Args:
            api_key: Optional API key override. Defaults to settings.TAVILY_API_KEY.
            endpoint: Optional REST endpoint URL override.
            timeout: Request timeout in seconds.
            client: Optional pre-existing httpx.AsyncClient instance.
            transport: Optional custom transport (e.g. httpx.MockTransport for tests).

        Raises:
            TavilyConfigError: If the API key is missing or empty.
        """
        resolved_key = api_key if api_key is not None else getattr(settings, "TAVILY_API_KEY", "")
        if not resolved_key or not isinstance(resolved_key, str) or not resolved_key.strip():
            raise TavilyConfigError(
                "Tavily API key is missing or empty. Please configure TAVILY_API_KEY."
            )

        self._api_key = resolved_key.strip()
        self.endpoint = endpoint or DEFAULT_TAVILY_ENDPOINT
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

    async def __aenter__(self) -> "TavilySearchClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client session if internally managed."""
        if self._manage_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        topic: str = "general",
        include_answer: bool = False,
        include_raw_content: bool = False,
    ) -> TavilySearchResponse:
        """
        Execute an asynchronous web research query via Tavily.

        Args:
            query: The research question or search terms.
            max_results: Maximum number of search results to return (must be >= 1).
            search_depth: 'basic' or 'advanced' depth.
            topic: 'general' or 'news'.
            include_answer: Whether to include an AI-generated concise answer.
            include_raw_content: Whether to include raw parsed web content.

        Returns:
            TavilySearchResponse containing normalized ResearchResult models.

        Raises:
            TavilyValidationError: If query is empty or max_results is invalid.
            TavilyTimeoutError: If the request times out.
            TavilyNetworkError: If a connection error occurs.
            TavilyHTTPError: If an HTTP 4xx/5xx status is returned.
            TavilyAPIError: If Tavily returns a provider error payload.
            TavilyParsingError: If response JSON is invalid or malformed.
        """
        # 1. Input Validation
        if not query or not isinstance(query, str) or not query.strip():
            raise TavilyValidationError("Search query must not be empty or whitespace only.")

        if not isinstance(max_results, int) or max_results < 1:
            raise TavilyValidationError(
                f"max_results must be a positive integer >= 1, got {max_results}."
            )

        cleaned_query = query.strip()

        # Build payload (never log the api_key)
        payload: Dict[str, Any] = {
            "api_key": self._api_key,
            "query": cleaned_query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        logger.info(
            "Starting Tavily search (query_len=%d, max_results=%d)",
            len(cleaned_query),
            max_results,
        )

        # 2. Execute HTTP POST
        try:
            response = await self._client.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            logger.warning("Tavily request timed out after %ss", self.timeout)
            raise TavilyTimeoutError(f"Tavily search request timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            logger.error("Tavily network failure: %s", exc)
            raise TavilyNetworkError(f"Network failure communicating with Tavily: {exc}") from exc

        # 3. Check HTTP Status
        if response.status_code >= 400:
            error_detail = "HTTP error"
            try:
                err_json = response.json()
                if isinstance(err_json, dict):
                    detail = err_json.get("detail") or err_json.get("error") or err_json.get("message")
                    if isinstance(detail, dict):
                        error_detail = detail.get("error") or detail.get("message") or str(detail)
                    elif detail:
                        error_detail = str(detail)
            except Exception:
                error_detail = response.text[:200] if response.text else "No error details returned"

            logger.error("Tavily returned HTTP %d: %s", response.status_code, error_detail)
            raise TavilyHTTPError(
                status_code=response.status_code,
                message=f"HTTP {response.status_code}: {error_detail}",
            )

        # 4. Parse JSON Response
        try:
            data = response.json()
        except Exception as exc:
            logger.error("Failed to parse Tavily JSON response: %s", exc)
            raise TavilyParsingError(f"Tavily response is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise TavilyParsingError("Tavily response payload must be a JSON object mapping.")

        # Check for provider error embedded in payload
        if "error" in data and data["error"]:
            err_msg = str(data["error"])
            logger.error("Tavily API returned error: %s", err_msg)
            raise TavilyAPIError(message=err_msg)

        # 5. Extract and Normalize Results
        raw_results = data.get("results")
        if raw_results is None or not isinstance(raw_results, list):
            raise TavilyParsingError("Tavily response is missing a valid 'results' list.")

        normalized_results: List[ResearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue

            raw_title = item.get("title")
            raw_url = item.get("url")
            raw_content = item.get("content")

            # Validate types of critical fields
            if raw_url is not None and not isinstance(raw_url, str):
                raise TavilyParsingError(f"Invalid URL type in result: {type(raw_url).__name__}")
            if raw_title is not None and not isinstance(raw_title, str):
                raise TavilyParsingError(f"Invalid title type in result: {type(raw_title).__name__}")

            score_val = item.get("score")
            normalized_score = float(score_val) if isinstance(score_val, (int, float)) else None

            normalized_results.append(
                ResearchResult(
                    title=raw_title or "",
                    url=raw_url or "",
                    content=raw_content if isinstance(raw_content, str) else "",
                    score=normalized_score,
                    published_date=item.get("published_date") if isinstance(item.get("published_date"), str) else None,
                    raw_content=item.get("raw_content") if isinstance(item.get("raw_content"), str) else None,
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"title", "url", "content", "score", "published_date", "raw_content"}
                    },
                )
            )

        if not normalized_results:
            logger.info("Tavily returned no results")
        else:
            logger.info("Tavily search completed (results=%d)", len(normalized_results))

        return TavilySearchResponse(
            success=True,
            query=cleaned_query,
            count=len(normalized_results),
            results=normalized_results,
            answer=data.get("answer") if isinstance(data.get("answer"), str) else None,
            response_time=data.get("response_time") if isinstance(data.get("response_time"), (int, float)) else None,
        )
