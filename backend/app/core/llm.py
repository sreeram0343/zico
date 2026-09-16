"""
Centralized OpenAI LLM provider module for ZICO.

This module provides a unified interface for instantiating and configuring
OpenAI chat models for ZICO agents, LangGraph nodes, and tools.

Architecture:
    app.core.config -> app.core.llm -> ZICO Agents -> LangGraph

Design Principles:
    - Centralization: All agents obtain their LLM through `get_chat_model()`.
    - Single Source of Truth: Reads credentials and model parameters exclusively
      from `app.core.config.settings`.
    - Security: API keys are never printed, logged, or exposed in exceptions.
    - Import Safety: Importing this module never triggers network requests or quota usage.
    - LangGraph & LangChain Compatibility: Returns standard `ChatOpenAI` instances
      ready for structured output and tool binding.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Default deterministic temperature for operations assistant
DEFAULT_TEMPERATURE: float = 0.0


class LLMConfigurationError(ValueError):
    """Raised when the LLM provider configuration is missing, invalid, or fails."""


def get_chat_model(
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    Construct and return a configured OpenAI Chat model instance.

    Uses a factory pattern to produce independent, thread-safe `ChatOpenAI`
    instances configured with credentials and model identifiers from application settings.

    Args:
        temperature: Optional sampling temperature override (defaults to 0.0 for determinism).
        model: Optional model identifier override (defaults to settings.OPENAI_MODEL).
        **kwargs: Additional parameters forwarded to the ChatOpenAI constructor.

    Returns:
        Configured `ChatOpenAI` instance ready for invocation or agent binding.

    Raises:
        LLMConfigurationError: If OPENAI_API_KEY or model name is missing, empty, or invalid.
    """
    # 1. Validate API Key
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        raise LLMConfigurationError(
            "OpenAI API key is missing or empty. Please set OPENAI_API_KEY in environment or .env."
        )

    # 2. Validate Model Name
    resolved_model = model if model is not None else getattr(settings, "OPENAI_MODEL", None)
    if not resolved_model or not isinstance(resolved_model, str) or not resolved_model.strip():
        raise LLMConfigurationError(
            "OpenAI model name is missing or empty. Please set OPENAI_MODEL in configuration."
        )

    # 3. Resolve Temperature
    if temperature is not None:
        resolved_temp = temperature
    else:
        resolved_temp = getattr(settings, "OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE)

    # Safe operational logging (never log credentials or prompts)
    logger.info("Initializing OpenAI chat model (model=%s, temperature=%s)", resolved_model, resolved_temp)

    # 4. Construct ChatOpenAI model
    try:
        return ChatOpenAI(
            model=resolved_model,
            api_key=api_key,
            temperature=resolved_temp,
            **kwargs,
        )
    except Exception as exc:
        err_msg = str(exc)
        # Redact the API key if the underlying exception string contains it
        if api_key and str(api_key) in err_msg:
            err_msg = err_msg.replace(str(api_key), "[REDACTED]")
        logger.error("Failed to initialize OpenAI chat model: %s", err_msg)
        raise LLMConfigurationError(f"Failed to initialize OpenAI chat model: {err_msg}") from exc
