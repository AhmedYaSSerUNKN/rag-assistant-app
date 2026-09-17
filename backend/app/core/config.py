"""Application settings, loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[2] == backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration. Nothing here is hard-coded at call sites."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "RAG Document Assistant API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Vector store (produced by notebooks/rag_pipeline.ipynb, copied into backend/data/)
    vector_store_dir: Path = BACKEND_DIR / "data" / "vector_store"
    collection_name: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 4
    min_relevance: float = 0.15  # cosine similarity floor; below this we refuse to answer
    max_context_chars: int = 6000

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: int = 120
    temperature: float = 0.0

    # CORS - comma separated list of allowed frontend origins
    allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed exactly once per process."""
    return Settings()
