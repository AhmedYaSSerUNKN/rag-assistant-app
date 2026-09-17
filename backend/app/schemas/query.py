"""Request / response models for the query API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question to answer from the indexed documents.",
        examples=["What is retrieval-augmented generation?"],
    )
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Override the number of chunks to retrieve."
    )


class SourceChunk(BaseModel):
    citation: str = Field(..., description="Human readable citation, e.g. 'paper.pdf (p. 3)'.")
    document: str
    page: int | None = None
    chunk_id: str
    score: float = Field(..., description="Cosine similarity between question and chunk.")
    preview: str = Field(..., description="First characters of the retrieved chunk.")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="Citation strings used in the answer.")
    contexts: list[SourceChunk] = Field(default_factory=list)
    grounded: bool = Field(..., description="False when no chunk passed the relevance floor.")
    latency_ms: int
    model: str


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    chunks_indexed: int
    embedding_model: str
    llm_model: str
    llm_reachable: bool
