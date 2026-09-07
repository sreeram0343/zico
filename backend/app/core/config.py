import os
from typing import Any, List, Literal, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Authoritative system configuration for ZICO Intelligent Travel Operations.
    Strictly validates environment variables and schema boundaries on startup.
    """

    PROJECT_NAME: str = "ZICO Intelligent Travel Operations"
    API_V1_STR: str = "/api/v1"
    APP_ENV: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Execution environment tier.",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application logging verbosity.",
    )

    # API Keys & External Providers
    OPENAI_API_KEY: str = Field(default="")
    SERPAPI_API_KEY: str = Field(default="")
    AVIATIONSTACK_API_KEY: str = Field(default="")
    TAVILY_API_KEY: str = Field(default="")
    DEFAULT_ORIGIN_IATA: str = Field(default="DAC")

    # PostgreSQL Database Settings
    POSTGRES_USER: str = Field(default="user")
    POSTGRES_PASSWORD: str = Field(default="pass")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535)
    POSTGRES_DB: str = Field(default="zico")
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/zico",
        description="SQLAlchemy async connection URI.",
    )

    # Redis Cache & Session State
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URI.",
    )

    # Qdrant Vector Database
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant service endpoint or ':memory:'.",
    )
    QDRANT_API_KEY: str = Field(default="")
    QDRANT_COLLECTION: str = Field(default="travel_policies")

    # Voice & Audio Services
    ELEVENLABS_API_KEY: str = Field(default="")
    ELEVENLABS_VOICE_ID: str = Field(default="21m00Tcm4TlvDq8ikWAM")

    # CORS Allowed Origins
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Field Validators
    # -----------------------------------------------------------------------

    @field_validator("APP_ENV", mode="before")
    @classmethod
    def validate_app_env(cls, v: Any) -> str:
        if isinstance(v, str):
            val = v.strip().lower()
            valid_envs = {"development", "staging", "production", "test"}
            if val not in valid_envs:
                raise ValueError(f"Invalid APP_ENV '{v}'. Must be one of {sorted(valid_envs)}")
            return val
        return v

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> str:
        if isinstance(v, str):
            val = v.strip().upper()
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if val not in valid_levels:
                raise ValueError(f"Invalid LOG_LEVEL '{v}'. Must be one of {sorted(valid_levels)}")
            return val
        return v

    @field_validator("DEFAULT_ORIGIN_IATA", mode="before")
    @classmethod
    def validate_origin_iata(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip().strip('"').strip("'").upper()
            if len(cleaned) != 3 or not cleaned.isalpha():
                raise ValueError(f"DEFAULT_ORIGIN_IATA must be a 3-letter alphabetic code, got '{v}'")
            return cleaned
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("DATABASE_URL must be a valid non-empty connection URI.")
        allowed_prefixes = (
            "postgresql+asyncpg://",
            "postgresql://",
            "sqlite+aiosqlite://",
            "sqlite://",
        )
        if not any(v.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(
                f"DATABASE_URL '{v}' has an unsupported scheme. Expected one of {allowed_prefixes}"
            )
        return v

    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("REDIS_URL must be a valid non-empty Redis connection URI.")
        if not (v.startswith("redis://") or v.startswith("rediss://")):
            raise ValueError(f"REDIS_URL '{v}' must start with redis:// or rediss://")
        return v

    @field_validator("QDRANT_URL")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("QDRANT_URL cannot be empty.")
        if v != ":memory:" and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"QDRANT_URL '{v}' must start with http://, https://, or be ':memory:'")
        return v

    # -----------------------------------------------------------------------
    # Helper Properties & Runtime Validation
    # -----------------------------------------------------------------------

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"

    @property
    def is_test(self) -> bool:
        return self.APP_ENV == "test"

    @property
    def sync_database_url(self) -> str:
        """Returns synchronous PostgreSQL URI for Alembic migrations and tooling."""
        if self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
        if self.DATABASE_URL.startswith("sqlite+aiosqlite://"):
            return self.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://", 1)
        return self.DATABASE_URL

    def validate_runtime(self) -> None:
        """
        Enforces strict production readiness constraints.
        Raises ValueError if production mode is missing critical secrets.
        """
        if self.is_production:
            missing: List[str] = []
            if not self.OPENAI_API_KEY:
                missing.append("OPENAI_API_KEY")
            if "localhost" in self.DATABASE_URL or "pass" in self.DATABASE_URL:
                missing.append("production DATABASE_URL with non-default credentials")
            if missing:
                raise ValueError(
                    f"Production startup validation failed. Missing required production configs: {', '.join(missing)}"
                )


settings = Settings()
