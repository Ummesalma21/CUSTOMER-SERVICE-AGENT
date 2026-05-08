from __future__ import annotations

from src.preference.score_candidates import score_answer
from src.utils.io import project_path, read_jsonl, write_json


def train_preference_ranker(config: dict) -> dict:
    pairs = read_jsonl(project_path("data", "processed", "preference_pairs.jsonl"))
    correct = sum(1 for p in pairs if score_answer(p["preferred"]) > score_answer(p["rejected"]))
    metrics = {"trained_pairs": len(pairs), "pair_accuracy": correct / len(pairs) if pairs else 0.0, "model": "rubric-ranker"}
    write_json(project_path("outputs", "preference", "model.json"), {"rubric": "citation+grounding+tool+concise"})
    write_json(project_path("outputs", "preference", "metrics.json"), metrics)
    return metrics

