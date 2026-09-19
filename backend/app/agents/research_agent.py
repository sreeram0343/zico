"""
Travel Research Agent for ZICO.

This module provides the specialized Research Agent responsible for:
    - Inspecting travel research requests from `TravelState`.
    - Constructing focused web search queries from user queries and state context.
    - Incorporating temporal constraints (e.g. 'current', 'latest', '2026') and destination context.
    - Targeting authoritative sources for visas, regulations, and airline policies.
    - Executing asynchronous search queries via `TavilySearchClient`.
    - Normalizing, deduplicating, and storing source-backed findings into `TravelState`.

Architecture:
    TravelState -> ResearchAgent -> LocationResolver (optional) -> TavilySearchClient -> Updated TravelState

Design Principles:
    - State Compliance: Strictly updates fields owned by research operations (`research_query`,
      `research_results`, `research_status`, `active_agent`, `completed_agents`, `errors`).
    - Intent Guard: Rejects non-research requests (`intent != 'research'`) without executing tools.
    - Deterministic First: Employs deterministic query construction by default, with optional
      structured LLM assistance for complex inquiries via `app.core.llm.get_chat_model`.
    - Provable Sources: Preserves result URLs, titles, contents, and scores for downstream validation.
    - Ambiguity & Error Resilient: Distinctly tracks 'success', 'no_results', and 'error'.
    - LangGraph Ready: Asynchronous entry point for workflow orchestration.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.logging import get_logger
from app.core.state import TravelState
from app.tools.location import LocationResolver, resolve_location
from app.tools.tavily_search import (
    ResearchResult,
    TavilySearchClient,
    TavilySearchError,
    TavilySearchResponse,
)

logger = get_logger(__name__)

# Default number of search results to request
DEFAULT_MAX_RESULTS = 5

# Keywords indicating a request for live or temporal information
TEMPORAL_KEYWORDS: Set[str] = {
    "current",
    "currently",
    "latest",
    "today",
    "2026",
    "recent",
    "updated",
    "this week",
    "now",
}

# Policy / regulation terms where authoritative source preference is advised
OFFICIAL_TOPICS: Set[str] = {
    "visa",
    "visas",
    "entry",
    "immigration",
    "passport",
    "customs",
    "baggage",
    "allowance",
    "restrictions",
    "regulations",
    "quarantine",
    "documents",
    "transit",
}

# Regex to remove common conversational conversational prefixes
CONVERSATIONAL_PREFIX_REGEX = re.compile(
    r"^(?:can you (?:please )?tell me|please tell me|i want to know|i would like to know|help me find|what are the|what is the|what do i need for|how do i get|can i get)\s+",
    re.IGNORECASE,
)

# System prompt for LLM-assisted query generation
RESEARCH_SYSTEM_PROMPT = """You are ZICO's Travel Research Query Generator.
Your task is to transform a travel inquiry into a focused, highly effective web search query.

Guidelines:
1. Extract the core subject, destination, policies, regulations, or requirements.
2. If the user asks for current, latest, or updated information, preserve that temporal intent (e.g. 'current', 'latest', '2026').
3. For visas, customs, baggage rules, or official regulations, prefer terms targeting authoritative sources (e.g. 'official requirements', 'rules').
4. Do NOT assume or invent traveler nationality, dates, or personal details not mentioned in the input.
5. Do NOT include search operators that restrict to non-existent domains.
6. Return only the structured schema with the clean query string."""


# ---------------------------------------------------------------------------
# Structured Output Model
# ---------------------------------------------------------------------------


class ResearchQueryDecision(BaseModel):
    """
    Structured web search query generated for travel research.
    """

    query: str = Field(
        description="Focused, concise web search query targeting authoritative travel and regulation sources."
    )


# ---------------------------------------------------------------------------
# Research Agent Implementation
# ---------------------------------------------------------------------------


class ResearchAgent:
    """
    Specialized agent for researching travel rules, visas, baggage, and destination policies.
    """

    def __init__(
        self,
        tavily_client: Optional[TavilySearchClient] = None,
        model: Optional[Any] = None,
        location_resolver: Optional[LocationResolver] = None,
        use_llm: Optional[bool] = None,
    ) -> None:
        """
        Initialize the Research Agent.

        Args:
            tavily_client: Optional pre-configured TavilySearchClient instance.
            model: Optional pre-configured chat model for query generation.
            location_resolver: Optional location resolver instance.
            use_llm: Optional boolean flag to force (True) or disable (False) LLM query generation.
                     If None, deterministic query generation is used by default, falling back
                     to LLM only when a model is explicitly provided.
        """
        self.tavily_client = tavily_client
        self._model = model
        self.location_resolver = location_resolver
        self.use_llm = use_llm

    def _resolve_location_name(self, location_query: Optional[str]) -> Optional[str]:
        """
        Safely resolve a city or country name via the location resolver.
        Falls back to the original string if resolution fails.
        """
        if not location_query or not isinstance(location_query, str) or not location_query.strip():
            return None

        cleaned = location_query.strip()
        try:
            if self.location_resolver:
                res = self.location_resolver.resolve(cleaned)
            else:
                res = resolve_location(cleaned)

            if res.status == "resolved" and res.city_name:
                return res.city_name
        except Exception as exc:
            logger.warning("Location resolution failed for %r: %s; using raw query", cleaned, exc)

        return cleaned

    def _generate_deterministic_query(self, user_query: str, state: TravelState) -> str:
        """
        Construct a focused search query deterministically from user query and state context.
        """
        query = user_query.strip()

        # 1. Strip trailing question marks and punctuation
        query = query.rstrip("?.! ")

        # 2. Check for multi-turn context (e.g. destination in state)
        raw_dest = state.get("destination")
        dest_name = self._resolve_location_name(raw_dest) if raw_dest else None

        if dest_name and dest_name.lower() not in query.lower():
            # Incorporate destination if not already present
            query = f"{dest_name} {query}"

        # 3. Check for temporal keywords in the user query
        has_temporal = any(w in user_query.lower() for w in TEMPORAL_KEYWORDS)
        if has_temporal and not any(w in query.lower() for w in TEMPORAL_KEYWORDS):
            query = f"current {query}"

        # 4. Remove conversational filler phrases
        cleaned = CONVERSATIONAL_PREFIX_REGEX.sub("", query).strip()
        if cleaned:
            query = cleaned

        # 5. Authoritative preference for regulatory/policy topics
        has_official_topic = any(topic in query.lower() for topic in OFFICIAL_TOPICS)
        if has_official_topic and "official" not in query.lower():
            query = f"{query} official"

        # 6. Normalize whitespace
        return re.sub(r"\s+", " ", query).strip()

    def _generate_llm_query(self, user_query: str, state: TravelState) -> Optional[str]:
        """
        Generate a structured research query using the centralized OpenAI LLM provider.
        Returns None if invocation fails or produces an invalid schema.
        """
        logger.info("Attempting LLM-assisted research query generation")
        try:
            model = self._model if self._model is not None else get_chat_model()
        except Exception as exc:
            logger.warning("Could not obtain chat model for research query generation: %s", exc)
            return None

        # Bind structured output schema
        if hasattr(model, "with_structured_output"):
            runnable = model.with_structured_output(ResearchQueryDecision)
        else:
            runnable = model

        # Provide context
        dest = state.get("destination")
        context_parts = [f"User inquiry: {user_query}"]
        if dest:
            context_parts.append(f"Destination context: {dest}")

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_parts)),
        ]

        try:
            response = runnable.invoke(messages)
        except Exception as exc:
            logger.warning("LLM query generation failed: %s", exc)
            return None

        # Parse structured decision
        decision: Optional[ResearchQueryDecision] = None
        if isinstance(response, ResearchQueryDecision):
            decision = response
        elif isinstance(response, dict):
            try:
                decision = ResearchQueryDecision(**response)
            except Exception:
                decision = None
        elif hasattr(response, "parsed"):
            parsed = getattr(response, "parsed")
            if isinstance(parsed, ResearchQueryDecision):
                decision = parsed
            elif isinstance(parsed, dict):
                try:
                    decision = ResearchQueryDecision(**parsed)
                except Exception:
                    decision = None

        if decision and decision.query and decision.query.strip():
            logger.info("LLM generated research query: %r", decision.query.strip())
            return decision.query.strip()

        logger.warning("LLM output was malformed or empty: %r", response)
        return None

    def _build_search_query(self, state: TravelState) -> Optional[str]:
        """
        Determine and construct the final search query to pass to Tavily.
        Returns None if insufficient information is present.
        """
        # 1. If research_query is already explicitly populated in state, reuse it
        existing_query = state.get("research_query")
        if existing_query and isinstance(existing_query, str) and existing_query.strip():
            return existing_query.strip()

        # 2. Check user_query
        user_query = state.get("user_query")
        if not user_query or not isinstance(user_query, str) or not user_query.strip():
            return None

        cleaned_user_query = user_query.strip()

        # 3. Determine whether to use LLM query construction
        should_use_llm = (self.use_llm is True) or (
            self.use_llm is None and self._model is not None
        )

        if should_use_llm:
            llm_query = self._generate_llm_query(cleaned_user_query, state)
            if llm_query:
                return llm_query
            logger.info("Falling back to deterministic query construction")

        # 4. Deterministic query generation
        return self._generate_deterministic_query(cleaned_user_query, state)

    def _deduplicate_results(self, results: List[ResearchResult]) -> List[Dict[str, Any]]:
        """
        Deduplicate search findings by normalized URL while preserving ranking order.
        """
        seen_urls: Set[str] = set()
        deduped: List[Dict[str, Any]] = []

        for item in results:
            raw_url = item.url or ""
            # Normalize URL by stripping trailing slashes and lowercasing
            normalized_url = raw_url.strip().rstrip("/").lower()

            if normalized_url and normalized_url in seen_urls:
                continue

            if normalized_url:
                seen_urls.add(normalized_url)

            deduped.append(item.model_dump() if hasattr(item, "model_dump") else dict(item))

        return deduped

    async def execute(self, state: TravelState) -> Dict[str, Any]:
        """
        Execute the travel research agent on the provided state.

        Args:
            state: Active TravelState dictionary.

        Returns:
            Partial state update dictionary containing updated `research_query`,
            `research_results`, `research_status`, `active_agent`, and `completed_agents`.
        """
        current_errors = list(state.get("errors", []))
        completed_agents = list(state.get("completed_agents", []))
        if "research_agent" not in completed_agents:
            completed_agents.append("research_agent")

        # 1. Intent Validation Guard
        intent = state.get("intent")
        if intent != "research":
            logger.warning("Research Agent invoked with non-research intent %r", intent)
            return {
                "research_status": "error",
                "active_agent": "research_agent",
                "completed_agents": completed_agents,
                "errors": current_errors
                + [f"Research Agent invoked with invalid intent '{intent}'. Expected 'research'."],
            }

        logger.info("Research Agent executing research workflow")

        # 2. Build Search Query
        search_query = self._build_search_query(state)
        if not search_query:
            msg = "Insufficient travel information to execute research: user_query is missing or empty."
            logger.error("Research Agent failed: %s", msg)
            return {
                "research_query": None,
                "research_results": [],
                "research_status": "error",
                "active_agent": "research_agent",
                "completed_agents": completed_agents,
                "errors": current_errors + [msg],
            }

        logger.info("Prepared research search query: %r", search_query)

        # 3. Query Tavily Search Tool
        client = self.tavily_client
        manage_client = False
        if client is None:
            client = TavilySearchClient()
            manage_client = True

        try:
            logger.info(
                "Executing Tavily search (query=%r, max_results=%d)",
                search_query,
                DEFAULT_MAX_RESULTS,
            )
            response: TavilySearchResponse = await client.search(
                query=search_query,
                max_results=DEFAULT_MAX_RESULTS,
            )
        except TavilySearchError as exc:
            err_msg = f"Tavily search provider error: {exc}"
            logger.error("Research Agent provider failure: %s", err_msg)
            return {
                "research_query": search_query,
                "research_results": [],
                "research_status": "error",
                "active_agent": "research_agent",
                "completed_agents": completed_agents,
                "errors": current_errors + [err_msg],
            }
        except Exception as exc:
            err_msg = f"Unexpected research execution error: {exc}"
            logger.error("Research Agent unexpected error: %s", err_msg)
            return {
                "research_query": search_query,
                "research_results": [],
                "research_status": "error",
                "active_agent": "research_agent",
                "completed_agents": completed_agents,
                "errors": current_errors + [err_msg],
            }
        finally:
            if manage_client:
                await client.close()

        # 4. Deduplicate and Normalize Results
        deduped_results = self._deduplicate_results(response.results)
        status = "success" if deduped_results else "no_results"

        logger.info(
            "Research Agent lookup completed (results=%d, status=%s)", len(deduped_results), status
        )

        return {
            "research_query": search_query,
            "research_results": deduped_results,
            "research_status": status,
            "active_agent": "research_agent",
            "completed_agents": completed_agents,
        }


# ---------------------------------------------------------------------------
# Module-level Convenience Entry Points
# ---------------------------------------------------------------------------


async def run_research_agent(
    state: TravelState,
    tavily_client: Optional[TavilySearchClient] = None,
    model: Optional[Any] = None,
    location_resolver: Optional[LocationResolver] = None,
    use_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Asynchronous entry point for ZICO Research Agent (LangGraph node compatible).

    Args:
        state: Active TravelState dictionary.
        tavily_client: Optional client override for testing or custom connection.
        model: Optional chat model override for query generation.
        location_resolver: Optional location resolver override.
        use_llm: Optional boolean flag to force or bypass LLM query generation.

    Returns:
        Partial state update dictionary.
    """
    agent = ResearchAgent(
        tavily_client=tavily_client,
        model=model,
        location_resolver=location_resolver,
        use_llm=use_llm,
    )
    return await agent.execute(state)
