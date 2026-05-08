from __future__ import annotations

from src.evaluation.evaluate_end_to_end import run_proposed


def answer(query: str, config: dict) -> dict:
    return run_proposed(query, config)

