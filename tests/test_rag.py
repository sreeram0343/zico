import pytest
from app.rag.service import (
    DEFAULT_POLICIES,
    PolicyDocument,
    PolicyRAGService,
    _compute_deterministic_embedding,
    get_rag_service,
)
from qdrant_client import QdrantClient


def test_compute_deterministic_embedding():
    """Verify deterministic embedding produces unit normalized vectors."""
    vec1 = _compute_deterministic_embedding("Flight cancellation refund rule")
    vec2 = _compute_deterministic_embedding("Flight cancellation refund rule")
    vec3 = _compute_deterministic_embedding("Completely unrelated luggage rules")

    assert len(vec1) == 1536
    assert vec1 == vec2
    assert vec1 != vec3

    # Check unit norm
    norm = sum(x * x for x in vec1)
    assert abs(norm - 1.0) < 1e-4


def test_rag_service_in_memory_indexing_and_search():
    """Verify PolicyRAGService indexing, seed policies, and search in memory."""
    client = QdrantClient(":memory:")
    service = PolicyRAGService(client=client, collection_name="test_travel_policies")

    count = service.seed_default_policies()
    assert count == len(DEFAULT_POLICIES)

    # Search for EU261 flight delay
    results = service.search_policies("EU 261 compensation flight delay", limit=2)
    assert len(results) > 0
    assert any("EU Regulation 261" in r["title"] or "EU" in r["content"] for r in results)

    # Test format context
    context_str = service.format_rag_context("What are the baggage size limits?")
    assert len(context_str) > 0
    assert "Policy" in context_str


def test_rag_service_custom_document_indexing():
    """Verify custom policy document indexing and retrieval."""
    client = QdrantClient(":memory:")
    service = PolicyRAGService(client=client, collection_name="test_custom_policies")

    custom_doc = PolicyDocument(
        id="policy_hotel_pet_rules",
        title="Hotel Pet Accommodation Protocol",
        category="HOTEL",
        content="Pets under 10kg are allowed with a $50 non-refundable sanitation fee.",
        metadata={"hotel": "Zico Partner Hotels"},
    )
    service.index_document(custom_doc)

    results = service.search_policies("pet fee hotel", limit=1)
    assert len(results) == 1
    assert results[0]["id"] == "policy_hotel_pet_rules"
    assert "sanitation fee" in results[0]["content"]


def test_singleton_get_rag_service():
    """Verify get_rag_service singleton initialization."""
    svc = get_rag_service()
    assert svc is not None
    assert isinstance(svc, PolicyRAGService)
