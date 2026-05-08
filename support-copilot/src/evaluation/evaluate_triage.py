from __future__ import annotations

from src.evaluation.metrics import classification_metrics
from src.triage.predict import predict_triage
from src.utils.io import project_path, read_jsonl


def evaluate_triage(config: dict) -> dict:
    rows = read_jsonl(project_path("data", "processed", "eval_set.jsonl"))
    preds = [{"decision": predict_triage(r["query"])["label"], "triage_margin": predict_triage(r["query"])["margin"]} for r in rows]
    return classification_metrics(rows, preds)

