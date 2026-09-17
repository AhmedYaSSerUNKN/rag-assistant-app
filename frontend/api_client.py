"""Thin wrapper around the backend API. No URL is hard-coded here."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))


class APIError(Exception):
    """Raised for any failure while talking to the backend."""


def _base_url() -> str:
    if not API_BASE_URL:
        raise APIError(
            "API_BASE_URL is not set. Copy .env.example to .env in the frontend/ "
            "folder, set the backend URL there, and restart the app."
        )
    return API_BASE_URL


def health() -> dict:
    try:
        response = requests.get(f"{_base_url()}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIError(f"Backend not reachable at {API_BASE_URL}: {exc}") from exc


def ask(question: str, top_k: int | None = None) -> dict:
    payload: dict = {"question": question}
    if top_k:
        payload["top_k"] = top_k
    try:
        response = requests.post(f"{_base_url()}/query", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.Timeout as exc:
        raise APIError("The request timed out. The local LLM may still be loading — try again.") from exc
    except requests.RequestException as exc:
        raise APIError(f"Could not reach the backend at {API_BASE_URL}: {exc}") from exc

    if response.status_code == 422:
        raise APIError("Invalid question: it must be between 3 and 1000 characters.")
    if response.status_code == 503:
        raise APIError("Backend is up but the vector store or LLM is not loaded. Check the server logs.")
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.reason
        raise APIError(f"Backend error {response.status_code}: {detail}")

    return response.json()
