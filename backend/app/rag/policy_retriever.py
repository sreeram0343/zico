import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.rag.service import (
    DEFAULT_POLICIES,
    EMBEDDING_DIMENSION,
    PolicyDocument,
    _compute_deterministic_embedding,
)

logger = logging.getLogger(__name__)


class AsyncPolicyRetriever:
    """
    Asynchronous Qdrant policy retrieval client for travel operations policies,
    regulations (EU261, US DOT, Schengen rules), and airline baggage limits.
    """

    def __init__(
        self,
        client: Optional[AsyncQdrantClient] = None,
        collection_name: Optional[str] = None,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        if client is not None:
            self.client = client
        else:
            # Initialize AsyncQdrantClient with repo settings and explicit timeout
            if settings.QDRANT_URL and settings.QDRANT_URL != ":memory:":
                self.client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
                    timeout=2.0,
                )
            else:
                self.client = AsyncQdrantClient(":memory:")

    async def ensure_collection(self) -> bool:
        """
        Ensures the travel policies collection is created with dense cosine distance.
        Returns True if successful, False on failure without raising exceptions.
        """
        try:
            collections_resp = await self.client.get_collections()
            existing_names = [c.name for c in collections_resp.collections]
            if self.collection_name not in existing_names:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
            return True
        except (TimeoutError, asyncio.TimeoutError) as exc:
            logger.warning(f"AsyncQdrantClient connection timed out during collection setup: {exc}")
            return False
        except Exception as exc:
            logger.warning(f"AsyncQdrantClient collection check failed gracefully: {exc}")
            return False

    async def embed_text(self, text: str) -> List[float]:
        """
        Generates dense vector embeddings using OpenAI Async client or deterministic fallback.
        """
        if (
            settings.OPENAI_API_KEY
            and not settings.OPENAI_API_KEY.startswith("test")
            and settings.APP_ENV != "test"
            and os.getenv("PYTEST_CURRENT_TEST") is None
        ):
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    timeout=3.0,
                    max_retries=1,
                )
                resp = await client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small",
                )
                return resp.data[0].embedding
            except Exception as exc:
                logger.warning(f"Async OpenAI embedding generation failed, falling back: {exc}")

        return _compute_deterministic_embedding(text, dim=EMBEDDING_DIMENSION)

    async def index_policy(self, doc: PolicyDocument) -> bool:
        """
        Asynchronously indexes a PolicyDocument into Qdrant.
        """
        try:
            await self.ensure_collection()
            embedding = await self.embed_text(f"{doc.title}\n{doc.content}")
            point_id = int(hashlib.md5(doc.id.encode("utf-8")).hexdigest()[:8], 16)

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "doc_id": doc.id,
                            "title": doc.title,
                            "category": doc.category,
                            "content": doc.content,
                            "metadata": doc.metadata,
                        },
                    )
                ],
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to asynchronously index policy document '{doc.id}': {exc}")
            return False

    async def seed_default_policies(self) -> int:
        """
        Seeds baseline travel policy documents into the async Qdrant collection.
        """
        success_count = 0
        for doc in DEFAULT_POLICIES:
            if await self.index_policy(doc):
                success_count += 1
        return success_count

    async def retrieve_policies(
        self,
        query: str,
        limit: int = 3,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves travel policy documents using semantic similarity search.

        TODO: Hybrid Retrieval (Dense + Sparse/BM25)
        Currently implements dense-only cosine retrieval. Once sparse/BM25 vectors (e.g., via FastEmbed
        or Qdrant sparse vectors) are provisioned in the cluster schema, combine dense cosine scores
        with sparse BM25 scores using reciprocal rank fusion (RRF) or Qdrant hybrid query.
        """
        try:
            await self.ensure_collection()
            query_vector = await self.embed_text(query)

            query_filter = None
            if category:
                query_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="category",
                            match=qmodels.MatchValue(value=category.upper()),
                        )
                    ]
                )

            # Perform async query
            try:
                results = await self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                )
                points = results.points
            except AttributeError:
                # Fallback to async search if query_points is not supported
                points = await self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                )

            formatted: List[Dict[str, Any]] = []
            for p in points:
                payload = p.payload or {}
                formatted.append({
                    "id": payload.get("doc_id", str(p.id)),
                    "title": payload.get("title", "Unknown Policy"),
                    "category": payload.get("category", "GENERAL"),
                    "content": payload.get("content", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": float(getattr(p, "score", 1.0)),
                })
            return formatted

        except (TimeoutError, asyncio.TimeoutError) as exc:
            logger.error(f"AsyncQdrantClient retrieval timeout for query '{query}': {exc}")
            return []
        except UnexpectedResponse as exc:
            logger.error(f"AsyncQdrantClient unexpected response during retrieval: {exc}")
            return []
        except Exception as exc:
            logger.error(f"AsyncQdrantClient retrieval error (non-fatal): {exc}")
            return []

    async def format_policy_context(self, query: str, limit: int = 3) -> str:
        """
        Asynchronously retrieves matching policies and formats them into a grounded context string.
        """
        matched = await self.retrieve_policies(query, limit=limit)
        if not matched:
            return "No specific travel policy documents found matching the inquiry."

        chunks: List[str] = []
        for i, item in enumerate(matched, 1):
            chunks.append(
                f"### [Policy {i}] {item['title']} (Category: {item['category']})\n"
                f"{item['content']}\n"
                f"*Source / Metadata*: {item['metadata']}"
            )
        return "\n\n".join(chunks)

    async def close(self) -> None:
        """Closes the async client connection."""
        try:
            await self.client.close()
        except Exception as exc:
            logger.debug(f"AsyncQdrantClient close notice: {exc}")


_async_retriever_instance: Optional[AsyncPolicyRetriever] = None


def get_async_policy_retriever() -> AsyncPolicyRetriever:
    """Returns singleton AsyncPolicyRetriever instance."""
    global _async_retriever_instance
    if _async_retriever_instance is None:
        _async_retriever_instance = AsyncPolicyRetriever()
    return _async_retriever_instance
