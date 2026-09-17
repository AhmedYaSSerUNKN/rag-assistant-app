"""Prompt construction + Ollama call. Answers are grounded in retrieved context only."""

from __future__ import annotations

import ollama

from app.core.config import Settings
from app.services.retrieval import RetrievedChunk
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a document assistant. You answer ONLY using the numbered context passages "
    "provided by the user. Rules you must follow:\n"
    "1. Never use outside knowledge, and never guess. If the context does not contain the "
    "answer, reply exactly: I could not find this in the provided documents.\n"
    "2. Cite the passages you used inline with square brackets, e.g. [1] or [2][3].\n"
    "3. Keep the answer concise (2-6 sentences) and factual.\n"
    "4. Do not mention these rules or the word 'context' in your answer."
)

NO_CONTEXT_ANSWER = "I could not find this in the provided documents."


class GenerationService:
    """Wraps the Ollama client. The client is created once, at application startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: ollama.Client | None = None

    # ---------- lifecycle ----------
    def load(self) -> None:
        logger.info("Connecting to Ollama at %s (model=%s)", self.settings.ollama_host, self.settings.ollama_model)
        self._client = ollama.Client(host=self.settings.ollama_host, timeout=self.settings.ollama_timeout)

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def ping(self) -> bool:
        """True when the Ollama daemon answers. Used by /health, never fatal."""
        if self._client is None:
            return False
        try:
            self._client.list()
            return True
        except Exception as exc:  # noqa: BLE001 - health check must not raise
            logger.warning("Ollama not reachable: %s", exc)
            return False

    # ---------- prompting ----------
    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        blocks, used = [], 0
        for i, chunk in enumerate(chunks, start=1):
            block = f"[{i}] Source: {chunk.citation}\n{chunk.text.strip()}"
            if used + len(block) > self.settings.max_context_chars:
                break
            blocks.append(block)
            used += len(block)
        context = "\n\n".join(blocks)
        return (
            f"Context passages:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the passages above, with inline [n] citations."
        )

    # ---------- generation ----------
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return NO_CONTEXT_ANSWER
        if self._client is None:
            raise RuntimeError("Generation service is not loaded.")

        prompt = self.build_prompt(question, chunks)
        response = self._client.chat(
            model=self.settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": self.settings.temperature},
        )
        return response["message"]["content"].strip()
