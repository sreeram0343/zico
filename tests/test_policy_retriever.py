from unittest.mock import AsyncMock, patch
import pytest
from qdrant_client import AsyncQdrantClient

from app.rag.policy_retriever import AsyncPolicyRetriever, get_async_policy_retriever
from app.rag.service import PolicyDocument


@pytest.mark.asyncio
async def test_async_policy_retriever_in_memory():
    """Verify in-memory async retrieval, seeding, and policy search (happy path)."""
    client = AsyncQdrantClient(":memory:")
    retriever = AsyncPolicyRetriever(client=client, collection_name="test_async_policies")

    seeded_count = await retriever.seed_default_policies()
    assert seeded_count > 0

    results = await retriever.retrieve_policies("flight cancellation 24 hour rule", limit=2)
    assert len(results) > 0
    assert any("US DOT" in r["title"] or "24-Hour" in r["title"] for r in results)

    context = await retriever.format_policy_context("baggage limits")
    assert "Policy" in context
    await retriever.close()


@pytest.mark.asyncio
async def test_async_policy_retriever_custom_document():
    """Verify indexing and retrieving custom policy documents."""
    client = AsyncQdrantClient(":memory:")
    retriever = AsyncPolicyRetriever(client=client, collection_name="test_custom_async")

    doc = PolicyDocument(
        id="policy_rail_transfer",
        title="Airport Express Rail Connection Policy",
        category="TRANSFER",
        content="Free airport express train transfers with valid boarding pass.",
        metadata={"operator": "Express Rail"},
    )
    indexed = await retriever.index_policy(doc)
    assert indexed is True

    results = await retriever.retrieve_policies("airport express train", limit=1)
    assert len(results) == 1
    assert results[0]["id"] == "policy_rail_transfer"
    await retriever.close()


@pytest.mark.asyncio
async def test_async_policy_retriever_timeout_error_resilience():
    """Verify that connection/timeout failures return empty results without crashing caller."""
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.get_collections.side_effect = TimeoutError("Connection timed out")

    retriever = AsyncPolicyRetriever(client=mock_client, collection_name="timeout_collection")

    # Should not raise exception
    results = await retriever.retrieve_policies("baggage allowance query")
    assert results == []

    context = await retriever.format_policy_context("visa rules")
    assert "No specific travel policy documents found" in context


@pytest.mark.asyncio
async def test_singleton_get_async_policy_retriever():
    """Verify singleton accessor."""
    instance = get_async_policy_retriever()
    assert isinstance(instance, AsyncPolicyRetriever)
