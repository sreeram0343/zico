import os
from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ZICO Intelligent Travel Operations"
    API_V1_STR: str = "/api/v1"
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    SERPAPI_API_KEY: str = Field(default=os.getenv("SERPAPI_API_KEY", ""))

    # Database & Redis Settings
    DATABASE_URL: str = Field(
        default=os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://user:pass@localhost:5432/zico",
        )
    )
    REDIS_URL: str = Field(
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

    # Qdrant Vector Database
    QDRANT_URL: str = Field(
        default=os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    QDRANT_API_KEY: str = Field(
        default=os.getenv("QDRANT_API_KEY", "")
    )
    QDRANT_COLLECTION: str = Field(
        default=os.getenv("QDRANT_COLLECTION", "travel_policies")
    )

    # Voice Services
    ELEVENLABS_API_KEY: str = Field(
        default=os.getenv("ELEVENLABS_API_KEY", "")
    )
    ELEVENLABS_VOICE_ID: str = Field(
        default=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    )

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


