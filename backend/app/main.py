"""FastAPI application entrypoint: lifespan startup loading, CORS, routes."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from app.utils.logging_config import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store, embedding model and LLM client ONCE at startup."""
    app.state.retrieval = RetrievalService(settings)
    app.state.generation = GenerationService(settings)

    try:
        app.state.retrieval.load()
    except Exception as exc:  # noqa: BLE001 - keep the app up so /health can report it
        logger.error("Could not load vector store: %s", exc)

    try:
        app.state.generation.load()
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not initialise Ollama client: %s", exc)

    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Retrieval-Augmented Generation API: retrieves document chunks and answers with citations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"name": settings.app_name, "docs": "/docs", "health": "/health"}
