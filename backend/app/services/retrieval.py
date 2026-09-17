"""Vector store loading and chunk retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    document: str
    page: int | None
    score: float

    @property
    def citation(self) -> str:
        return f"{self.document} (p. {self.page})" if self.page else self.document


class RetrievalService:
    """Loads the persisted Chroma store + embedding model once, at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._collection = None
        self._model = None
        self.store_config: dict = {}

    # ---------- lifecycle ----------
    def load(self) -> None:
        # Heavy imports happen here, not at module import time, so the API tests
        # (which override these services) run without torch/chroma installed.
        import chromadb
        from sentence_transformers import SentenceTransformer

        store_dir = Path(self.settings.vector_store_dir)
        if not store_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {store_dir}. "
                "Run notebooks/rag_pipeline.ipynb first (section 2.7 exports it here)."
            )

        config_path = store_dir / "store_config.json"
        if config_path.exists():
            self.store_config = json.loads(config_path.read_text(encoding="utf-8"))

        embedding_model = self.store_config.get("embedding_model", self.settings.embedding_model)
        if embedding_model != self.settings.embedding_model:
            logger.warning(
                "EMBEDDING_MODEL in .env (%s) differs from the model used to build the store (%s). "
                "Using the store's model to keep vectors compatible.",
                self.settings.embedding_model,
                embedding_model,
            )

        collection_name = self.store_config.get("collection_name", self.settings.collection_name)

        logger.info("Loading Chroma store from %s", store_dir)
        client = chromadb.PersistentClient(path=str(store_dir))
        self._collection = client.get_collection(name=collection_name)

        logger.info("Loading embedding model %s", embedding_model)
        self._model = SentenceTransformer(embedding_model)

        logger.info("Vector store ready: %d chunks indexed", self.count())

    @property
    def is_loaded(self) -> bool:
        return self._collection is not None and self._model is not None

    def count(self) -> int:
        return self._collection.count() if self._collection is not None else 0

    def embedding_model_name(self) -> str:
        return self.store_config.get("embedding_model", self.settings.embedding_model)

    # ---------- retrieval ----------
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.is_loaded:
            raise RuntimeError("Retrieval service is not loaded.")

        k = top_k or self.settings.top_k
        embedding = self._model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for chunk_id, text, meta, distance in zip(ids, docs, metas, dists):
            meta = meta or {}
            page = meta.get("page")
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=text or "",
                    document=str(meta.get("source", "unknown")),
                    page=int(page) if page is not None else None,
                    # Chroma cosine distance -> similarity
                    score=round(1.0 - float(distance), 4),
                )
            )
        return chunks
