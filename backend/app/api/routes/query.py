"""API routes: GET /health and POST /query."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, SourceChunk
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from app.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


# --- dependencies (overridable in tests) -------------------------------------
def get_retrieval_service(request: Request) -> RetrievalService:
    service = getattr(request.app.state, "retrieval", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not loaded. See server logs.",
        )
    return service


def get_generation_service(request: Request) -> GenerationService:
    service = getattr(request.app.state, "generation", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not loaded. See server logs.",
        )
    return service


# --- routes ------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    retrieval: RetrievalService | None = getattr(request.app.state, "retrieval", None)
    generation: GenerationService | None = getattr(request.app.state, "generation", None)

    loaded = bool(retrieval and retrieval.is_loaded)
    llm_ok = bool(generation and generation.ping())

    return HealthResponse(
        status="ok" if (loaded and llm_ok) else "degraded",
        vector_store_loaded=loaded,
        chunks_indexed=retrieval.count() if loaded else 0,
        embedding_model=retrieval.embedding_model_name() if loaded else settings.embedding_model,
        llm_model=settings.ollama_model,
        llm_reachable=llm_ok,
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    retrieval: RetrievalService = Depends(get_retrieval_service),
    generation: GenerationService = Depends(get_generation_service),
) -> QueryResponse:
    started = time.perf_counter()
    question = payload.question.strip()

    chunks = retrieval.retrieve(question, top_k=payload.top_k)
    relevant = [c for c in chunks if c.score >= settings.min_relevance]
    logger.info("question=%r retrieved=%d relevant=%d", question, len(chunks), len(relevant))

    try:
        answer = generation.generate(question, relevant)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    return QueryResponse(
        answer=answer,
        sources=list(dict.fromkeys(c.citation for c in relevant)),
        contexts=[
            SourceChunk(
                citation=c.citation,
                document=c.document,
                page=c.page,
                chunk_id=c.chunk_id,
                score=c.score,
                preview=c.text[:300].strip(),
            )
            for c in relevant
        ],
        grounded=bool(relevant),
        latency_ms=int((time.perf_counter() - started) * 1000),
        model=settings.ollama_model,
    )
