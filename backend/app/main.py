from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings
from app.db.models import Base
from app.db.redis import close_redis, init_redis
from app.db.session import engine
from app.rag.service import get_rag_service

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("zico.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager handling database initialization,
    Redis pool connection, and Policy RAG vector indexing.
    """
    logger.info("Initializing ZICO Intelligent Travel Operations Backend...")

    # 1. Initialize Database Tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL database tables initialized.")
    except Exception as exc:
        logger.warning(f"Database table initialization notice: {exc}")

    # 2. Initialize Redis Connection Pool
    try:
        await init_redis()
        logger.info("Redis cache and session pool initialized.")
    except Exception as exc:
        logger.warning(f"Redis initialization notice: {exc}")

    # 3. Seed Policy RAG Vector Store
    try:
        rag = get_rag_service()
        count = rag.seed_default_policies()
        logger.info(f"Qdrant policy vector store initialized with {count} baseline travel policies.")
    except Exception as exc:
        logger.warning(f"Policy RAG seed notice: {exc}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down ZICO backend services...")
    try:
        await close_redis()
    except Exception as exc:
        logger.error(f"Error during Redis shutdown: {exc}")
    try:
        await engine.dispose()
    except Exception as exc:
        logger.error(f"Error during database engine shutdown: {exc}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root entrypoint returning system metadata."""
    return {
        "system": settings.PROJECT_NAME,
        "version": "1.0.0",
        "phase": "Phase 1 - Core Multi-Agent Travel Operations",
        "docs": f"{settings.API_V1_STR}/docs",
        "status": "operational",
    }
