from __future__ import annotations

from src.retrieval.search_kb import build_index


def build_faiss_index() -> dict:
    """Build a JSON index and, when using a real retriever, a FAISS sidecar."""
    return build_index()
