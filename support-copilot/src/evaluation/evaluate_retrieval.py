from __future__ import annotations

from src.evaluation.evaluate_end_to_end import run_baseline
from src.evaluation.metrics import retrieval_metrics
from src.utils.io import project_path, read_jsonl


def evaluate_retrieval(config: dict) -> dict:
    rows = read_jsonl(project_path("data", "processed", "eval_set.jsonl"))
    preds = [run_baseline(r["query"], config) for r in rows]
    return retrieval_metrics(rows, preds, k=5)

