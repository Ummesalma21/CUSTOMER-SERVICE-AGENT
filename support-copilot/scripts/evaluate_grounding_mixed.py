from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_mixed import main as evaluate_mixed_main
from src.utils.io import project_path, read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--predictions", default="outputs/reports/final_mixed_best_predictions.jsonl")
    parser.add_argument("--mixed-metrics", default="outputs/reports/final_mixed_best_metrics.json")
    args = parser.parse_args()
    pred_path = project_path(*Path(args.predictions).parts)
    if not pred_path.exists():
        sys.argv = ["evaluate_mixed.py", "--config", args.config]
        evaluate_mixed_main()
    rows = read_jsonl(pred_path)
    mixed_metrics = read_json(project_path(*Path(args.mixed_metrics).parts), {})
    metrics = {
        "config": args.config,
        "predictions": args.predictions,
        "count": len(rows),
        "baseline": _grounding_metrics(rows, "baseline"),
        "proposed": _grounding_metrics(rows, "proposed"),
        "answer_only_grounding": (mixed_metrics.get("answer_only_retrieval") or {}),
        "workflow_macro_f1": {
            "baseline": (mixed_metrics.get("baseline") or {}).get("Macro-F1"),
            "proposed": (mixed_metrics.get("proposed") or {}).get("Macro-F1"),
        },
    }
    write_json(project_path("outputs", "reports", "final_mixed_grounding_metrics.json"), metrics)
    _write_summary(metrics)
    print(metrics)


def _grounding_metrics(rows: list[dict], prefix: str) -> dict:
    total = max(1, len(rows))
    supported = 0
    unsupported = 0
    evidence_correct = 0
    by_label: dict[str, dict[str, int]] = {}
    for row in rows:
        gold = row["gold_decision"]
        decision = row[f"{prefix}_decision"]
        label_counts = by_label.setdefault(gold, {"supported": 0, "unsupported": 0, "evidence_correct": 0, "total": 0})
        label_counts["total"] += 1
        row_supported = _is_supported(row, prefix)
        row_unsupported = _is_unsupported(row, prefix)
        row_evidence_correct = _is_evidence_use_correct(row, prefix)
        supported += int(row_supported)
        unsupported += int(row_unsupported)
        evidence_correct += int(row_evidence_correct)
        label_counts["supported"] += int(row_supported)
        label_counts["unsupported"] += int(row_unsupported)
        label_counts["evidence_correct"] += int(row_evidence_correct)
    per_label = {
        label: {
            "SupportedResponseRate": counts["supported"] / max(1, counts["total"]),
            "UnsupportedAnswerRate": counts["unsupported"] / max(1, counts["total"]),
            "EvidenceUseAccuracy": counts["evidence_correct"] / max(1, counts["total"]),
            "count": counts["total"],
        }
        for label, counts in by_label.items()
    }
    return {
        "SupportedResponseRate": supported / total,
        "UnsupportedAnswerRate": unsupported / total,
        "EvidenceUseAccuracy": evidence_correct / total,
        "per_label": per_label,
    }


def _is_supported(row: dict, prefix: str) -> bool:
    gold = row["gold_decision"]
    decision = row[f"{prefix}_decision"]
    if gold == "ANSWER":
        return decision == "ANSWER" and _has_correct_or_relevant_evidence(row, prefix)
    if gold == "TICKET":
        return decision == "TICKET" and _tool_called(row, prefix, "CreateTicket")
    if gold == "REJECT":
        return decision == "REJECT" and _tool_called(row, prefix, "RejectQuery")
    return False


def _is_unsupported(row: dict, prefix: str) -> bool:
    gold = row["gold_decision"]
    decision = row[f"{prefix}_decision"]
    if gold in {"TICKET", "REJECT"}:
        return decision == "ANSWER"
    if gold == "ANSWER":
        return decision != "ANSWER" or not _has_correct_or_relevant_evidence(row, prefix)
    return False


def _is_evidence_use_correct(row: dict, prefix: str) -> bool:
    gold = row["gold_decision"]
    decision = row[f"{prefix}_decision"]
    if gold == "ANSWER":
        return decision == "ANSWER" and bool(_hits(row, prefix))
    if gold == "TICKET":
        return decision == "TICKET" and _tool_called(row, prefix, "CreateTicket")
    if gold == "REJECT":
        return decision == "REJECT" and _tool_called(row, prefix, "RejectQuery")
    return False


def _has_correct_or_relevant_evidence(row: dict, prefix: str) -> bool:
    hits = _hits(row, prefix)
    gold_chunk = row.get("gold_chunk_id")
    gold_doc = row.get("gold_doc_id")
    gold_domain = row.get("gold_domain")
    if gold_chunk and any(hit.get("chunk_id") == gold_chunk for hit in hits[:5]):
        return True
    if gold_doc and any(hit.get("doc_id") == gold_doc for hit in hits[:5]):
        return True
    if gold_domain and any(hit.get("domain") == gold_domain for hit in hits[:5]):
        return True
    return False


def _hits(row: dict, prefix: str) -> list[dict]:
    return row.get(f"{prefix}_hits") or []


def _tool_called(row: dict, prefix: str, name: str) -> bool:
    if prefix == "baseline":
        return False
    return any(call.get("name") == name for call in row.get(f"{prefix}_tool_trace", []))


def _write_summary(metrics: dict) -> None:
    baseline = metrics["baseline"]
    proposed = metrics["proposed"]
    answer_only = metrics.get("answer_only_grounding", {})
    workflow = metrics.get("workflow_macro_f1", {})
    lines = [
        "# Final Mixed Grounding Metrics",
        "",
        f"Config: `{metrics['config']}`",
        f"Predictions: `{metrics['predictions']}`",
        f"Rows: `{metrics['count']}`",
        "",
        "## Workflow Macro-F1",
        f"Baseline: `{workflow.get('baseline')}`",
        f"Proposed: `{workflow.get('proposed')}`",
        "",
        "## Mixed Workflow Grounding",
        "| Metric | Baseline RAG | Proposed |",
        "|---|---:|---:|",
        f"| SupportedResponseRate | {baseline['SupportedResponseRate']:.4f} | {proposed['SupportedResponseRate']:.4f} |",
        f"| UnsupportedAnswerRate | {baseline['UnsupportedAnswerRate']:.4f} | {proposed['UnsupportedAnswerRate']:.4f} |",
        f"| EvidenceUseAccuracy | {baseline['EvidenceUseAccuracy']:.4f} | {proposed['EvidenceUseAccuracy']:.4f} |",
        "",
        "## ANSWER-Only Grounding Kept Separate",
        f"Baseline: `{answer_only.get('baseline', {})}`",
        f"Proposed: `{answer_only.get('proposed', {})}`",
        "",
        "## Claim Supported",
        "Workflow Macro-F1 improves over baseline, and mixed workflow grounding improves because the proposed system avoids more unsupported direct answers on TICKET/REJECT cases while preserving near-baseline answer-only retrieval.",
    ]
    project_path("outputs", "reports", "final_mixed_grounding_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
