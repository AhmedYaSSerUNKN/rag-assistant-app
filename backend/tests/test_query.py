"""API tests. They use fake services via dependency overrides, so pytest passes
without Ollama running and without a built vector store (CI-friendly)."""

import pytest
from fastapi.testclient import TestClient

from app.api.routes.query import get_generation_service, get_retrieval_service
from app.main import app
from app.services.retrieval import RetrievedChunk


class FakeRetrieval:
    is_loaded = True

    def count(self) -> int:
        return 3

    def embedding_model_name(self) -> str:
        return "fake-embedding-model"

    def retrieve(self, question: str, top_k=None):
        return [
            RetrievedChunk(
                chunk_id="doc1::0",
                text="RAG combines a retriever over a document corpus with a generator LLM.",
                document="rag_intro.pdf",
                page=2,
                score=0.81,
            )
        ]


class FakeGeneration:
    is_loaded = True

    def ping(self) -> bool:
        return True

    def generate(self, question, chunks):
        return "Retrieval-augmented generation grounds the LLM in retrieved passages [1]."


@pytest.fixture
def client():
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrieval()
    app.dependency_overrides[get_generation_service] = lambda: FakeGeneration()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_query_happy_path(client):
    """Happy path: a valid question returns a grounded answer with citations."""
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"]
    assert body["grounded"] is True
    assert body["sources"] == ["rag_intro.pdf (p. 2)"]
    assert body["contexts"][0]["chunk_id"] == "doc1::0"
    assert body["latency_ms"] >= 0


def test_query_invalid_input_returns_422(client):
    """Invalid input: an empty question fails Pydantic validation with 422."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422

    assert client.post("/query", json={}).status_code == 422
    assert client.post("/query", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["vector_store_loaded"] in (True, False)
