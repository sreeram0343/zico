import hashlib
import logging
import math
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536


class PolicyDocument(BaseModel):
    """Represents a travel policy, regulation, or guideline document."""

    id: str
    title: str
    category: str = Field(
        description="Category such as CANCELLATION, BAGGAGE, COMPENSATION, VISA, INSURANCE"
    )
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Default foundational policies for traveler support
DEFAULT_POLICIES: List[PolicyDocument] = [
    PolicyDocument(
        id="policy_eu261_compensation",
        title="EU Regulation 261/2004 Flight Delay & Cancellation Rights",
        category="COMPENSATION",
        content=(
            "Under EU 261/2004, air passengers are entitled to fixed financial compensation for flight cancellations "
            "or delays exceeding 3 hours at final destination, unless caused by extraordinary circumstances. "
            "Compensation tiers: €250 for flights up to 1,500 km; €400 for intra-EU flights > 1,500 km and other flights "
            "between 1,500 km and 3,500 km; €600 for all other flights > 3,500 km. "
            "Airlines must also provide 'duty of care' including complimentary meals, refreshments, and hotel accommodation "
            "if an overnight stay becomes necessary."
        ),
        metadata={"jurisdiction": "EU", "regulation": "EU 261/2004", "type": "Statutory Rights"},
    ),
    PolicyDocument(
        id="policy_us_dot_24hr_cancellation",
        title="US DOT 24-Hour Flight Cancellation & Full Refund Rule",
        category="CANCELLATION",
        content=(
            "Under US Department of Transportation (DOT) federal regulations (14 CFR 259.5(b)(4)), all airlines "
            "operating flights to, from, or within the United States must allow passengers to cancel their booking "
            "and receive a 100% full refund without penalty if the reservation is cancelled within 24 hours of purchase, "
            "provided the booking was made at least 7 days (168 hours) prior to the scheduled departure time."
        ),
        metadata={"jurisdiction": "US", "authority": "US DOT", "regulation": "14 CFR 259.5"},
    ),
    PolicyDocument(
        id="policy_baggage_allowance_limits",
        title="Standard Airline Baggage Allowance and Excess Weight Guidelines",
        category="BAGGAGE",
        content=(
            "Standard economy class baggage policies generally permit one carry-on bag (maximum dimensions 56 x 36 x 23 cm, "
            "max weight 7 kg / 15 lbs) plus one personal item (laptop bag/purse) stowed under the seat. "
            "Checked baggage allowance is typically 1 piece up to 23 kg (50 lbs) for transatlantic/international itineraries. "
            "Overweight bags between 23 kg and 32 kg incur an excess fee (typically $100-$150 USD). "
            "No single checked bag exceeding 32 kg (70 lbs) is accepted for carriage due to airport safety regulations."
        ),
        metadata={"standard": "IATA Guidelines", "category": "Luggage"},
    ),
    PolicyDocument(
        id="policy_schengen_passport_validity",
        title="Schengen Area 6-Month Passport Validity and 90/180 Day Visa Rule",
        category="VISA",
        content=(
            "Travelers entering the European Schengen Area from visa-exempt non-EU countries must hold a passport "
            "valid for at least 3 months beyond the intended date of departure from the Schengen territory, issued within the past 10 years. "
            "Many airlines enforce a 6-month validity rule upon departure to avoid boarding denial. "
            "Non-EU short-stay visitors are restricted to a maximum stay of 90 days within any rolling 180-day period."
        ),
        metadata={"jurisdiction": "Schengen Zone", "category": "Immigration"},
    ),
    PolicyDocument(
        id="policy_travel_insurance_interruption",
        title="Travel Insurance Trip Interruption and Emergency Medical Guidelines",
        category="INSURANCE",
        content=(
            "Trip interruption coverage reimburses the non-refundable, prepaid expenses of a trip if you must cut your journey short "
            "due to a covered emergency (e.g., serious unforeseen illness, injury, or natural disaster rendering accommodation uninhabitable). "
            "In case of medical emergency abroad, travelers must contact their insurer's 24/7 assistance hotline prior to receiving non-urgent treatment. "
            "Receipts, medical practitioner reports, and airline delay confirmation letters must be retained to file formal claims."
        ),
        metadata={"category": "Insurance", "claim_window_days": 30},
    ),
]


def _compute_deterministic_embedding(text: str, dim: int = EMBEDDING_DIMENSION) -> List[float]:
    """
    Generates a deterministic normalized pseudo-embedding vector for offline / testing environments.
    Uses SHA-512 hashes distributed across dimensions with unit L2-norm normalization.
    """
    vec = [0.0] * dim
    words = text.lower().split()
    if not words:
        words = ["empty"]

    for word in words:
        h = hashlib.sha512(word.encode("utf-8")).digest()
        for i in range(min(dim, len(h) * 4)):
            byte_val = h[i % len(h)]
            pos = (i * 37) % dim
            vec[pos] += (byte_val - 128) / 128.0

    # Normalize vector to unit length
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0
    return vec


class PolicyRAGService:
    """Manages vector embeddings, indexing, and semantic search for travel policies in Qdrant."""

    def __init__(self, client: Optional[QdrantClient] = None, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        if client is not None:
            self.client = client
        else:
            # Fast probe to check if remote Qdrant is reachable
            is_reachable = False
            if settings.QDRANT_URL and settings.QDRANT_URL != ":memory:":
                try:
                    import urllib.request
                    req = urllib.request.Request(
                        f"{settings.QDRANT_URL.rstrip('/')}/collections",
                        headers={"User-Agent": "ZicoRAG/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=0.3) as response:
                        if response.status == 200:
                            is_reachable = True
                except Exception:
                    is_reachable = False

            if is_reachable:
                try:
                    self.client = QdrantClient(
                        url=settings.QDRANT_URL,
                        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
                        timeout=1.0,
                    )
                except Exception:
                    self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(":memory:")

        self._ensure_collection()


    def _ensure_collection(self) -> None:
        """Initializes the vector collection if it does not already exist."""
        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]
            if self.collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            logger.error(f"Error ensuring Qdrant collection: {exc}")

    def embed_text(self, text: str) -> List[float]:
        """Generates embedding vector using OpenAI or deterministic fallback."""
        if (
            settings.OPENAI_API_KEY
            and not settings.OPENAI_API_KEY.startswith("test")
            and settings.APP_ENV != "test"
            and os.getenv("PYTEST_CURRENT_TEST") is None
        ):
            try:
                from openai import OpenAI
                oai = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=3.0, max_retries=1)
                resp = oai.embeddings.create(
                    input=text,
                    model="text-embedding-3-small",
                )
                return resp.data[0].embedding
            except Exception as exc:
                logger.warning(f"OpenAI embedding generation failed, falling back to deterministic: {exc}")
        return _compute_deterministic_embedding(text, dim=EMBEDDING_DIMENSION)


    def index_document(self, doc: PolicyDocument) -> None:
        """Indexes a PolicyDocument into Qdrant."""
        embedding = self.embed_text(f"{doc.title}\n{doc.content}")
        # Generate stable numerical or UUID id for point
        point_id = int(hashlib.md5(doc.id.encode("utf-8")).hexdigest()[:8], 16)

        self.client.upsert(
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

    def seed_default_policies(self) -> int:
        """Indexes the built-in foundational travel policies."""
        for doc in DEFAULT_POLICIES:
            self.index_document(doc)
        return len(DEFAULT_POLICIES)

    def search_policies(
        self,
        query: str,
        limit: int = 3,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic similarity search on travel policy collection.
        Returns matched documents with similarity score and metadata.
        """
        query_vector = self.embed_text(query)

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

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
            points = results.points
        except Exception:
            try:
                # Older / alternative qdrant search method fallback
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                )
            except Exception as exc:
                logger.error(f"Failed to query Qdrant points: {exc}")
                return []

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

    def format_rag_context(self, query: str, limit: int = 3) -> str:
        """Retrieves matching policies and formats into a grounded markdown context snippet."""
        matched = self.search_policies(query, limit=limit)
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


# Global singleton instance
_rag_service_instance: Optional[PolicyRAGService] = None


def get_rag_service() -> PolicyRAGService:
    """Returns or initializes the singleton PolicyRAGService."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = PolicyRAGService()
        _rag_service_instance.seed_default_policies()
    return _rag_service_instance
