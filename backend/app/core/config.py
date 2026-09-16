"""
Application configuration module for the ZICO project.

Centralizes application configuration and environment variables using
pydantic-settings. Automatically reads configuration from system environment
variables and an optional .env file with sensible defaults for local development.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings and environment variable schema for ZICO.

    Attributes:
        OPENAI_API_KEY: API key for OpenAI model services.
        TAVILY_API_KEY: API key for Tavily search operations.
        AVIATIONSTACK_API_KEY: API key for Aviationstack flight status and schedules.
        APP_ENV: Current application environment (e.g. development, staging, production, test).
        LOG_LEVEL: Logging severity level (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    # Application Environment & Logging Defaults
    APP_ENV: str = Field(
        default="development",
        description="Application runtime environment.",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application logging verbosity level.",
    )

    # External Provider API Keys (never hardcoded; loaded via env or .env)
    OPENAI_API_KEY: str = Field(
        default="",
        description="API key for OpenAI API services.",
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier for travel operations.",
    )
    TAVILY_API_KEY: str = Field(
        default="",
        description="API key for Tavily search API.",
    )
    AVIATIONSTACK_API_KEY: str = Field(
        default="",
        description="API key for Aviationstack flight tracking API.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Expose a single singleton settings instance for application-wide consumption
settings = Settings()
