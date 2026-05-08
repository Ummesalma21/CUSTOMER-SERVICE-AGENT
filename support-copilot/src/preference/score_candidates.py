from __future__ import annotations


def score_answer(answer: str, expected_tool: str | None = None) -> float:
    score = 0.0
    if "[doc_id=" in answer and "chunk_id=" in answer and "span=" in answer:
        score += 3.0
    if "I could not find enough KB evidence" in answer:
        score += 1.0
    if "I can only help" in answer:
        score += 1.0
    if len(answer.split()) <= 80:
        score += 1.0
    if expected_tool == "ANSWER" and "[doc_id=" in answer:
        score += 1.0
    return score


def choose_best(candidates: list[str], expected_tool: str | None = None) -> str:
    return max(candidates, key=lambda a: score_answer(a, expected_tool)) if candidates else ""

