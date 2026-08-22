from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.rag.service import PolicyDocument, get_rag_service

router = APIRouter()


class RAGQueryRequest(BaseModel):
    query: str = Field(description="Policy search query")
    limit: int = Field(default=3, ge=1, le=10)
    category: Optional[str] = Field(default=None, description="Optional category filter")


class RAGQueryResponse(BaseModel):
    query: str
    context: str
    results: List[Dict[str, Any]]


class IndexDocumentResponse(BaseModel):
    id: str
    status: str
    message: str


@router.post("/query", response_model=RAGQueryResponse)
async def query_policies(payload: RAGQueryRequest) -> RAGQueryResponse:
    """
    Queries travel policy vector database and returns matched documents and grounded context.
    """
    rag = get_rag_service()
    results = rag.search_policies(
        query=payload.query,
        limit=payload.limit,
        category=payload.category,
    )
    context = rag.format_rag_context(query=payload.query, limit=payload.limit)

    return RAGQueryResponse(
        query=payload.query,
        context=context,
        results=results,
    )


@router.post("/index", response_model=IndexDocumentResponse)
async def index_custom_policy(doc: PolicyDocument) -> IndexDocumentResponse:
    """
    Indexes a new travel policy document into Qdrant.
    """
    rag = get_rag_service()
    rag.index_document(doc)

    return IndexDocumentResponse(
        id=doc.id,
        status="SUCCESS",
        message=f"Document '{doc.title}' successfully indexed into vector database.",
    )
