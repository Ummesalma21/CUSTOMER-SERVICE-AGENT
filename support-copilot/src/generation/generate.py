from __future__ import annotations

from src.generation.templates import cited_answer


def generate_answer(query: str, passages: list[dict]) -> str:
    return cited_answer(query, passages)

