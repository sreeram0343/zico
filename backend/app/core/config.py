import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

