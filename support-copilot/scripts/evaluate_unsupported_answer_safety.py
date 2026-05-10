from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import project_path, read_jsonl, write_json, write_jsonl


def main() -> None:
    rows = _load_rows()
    metrics, predictions = compute_safety(rows)
    write_json(project_path("outputs", "reports", "unsupported_answer_safety_metrics.json"), metrics)
    write_jsonl(project_path("outputs", "reports", "unsupported_answer_safety_predictions.jsonl"), predictions)
    _write_summary(metrics)
    print(json.dumps(metrics, indent=2))


def _load_rows() -> list[dict]:
    path = project_path("outputs", "reports", "three_way_mixed_predictions.jsonl")
    if not path.exists():
        raise FileNotFoundError("Run scripts/evaluate_three_way_final.py first to create three_way_mixed_predictions.jsonl")
    return read_jsonl(path)


def compute_safety(rows: list[dict]) -> tuple[dict, list[dict]]:
    unsupported = [r for r in rows if r.get("gold_decision") in {"TICKET", "REJECT"}]
    answerable = [r for r in rows if r.get("gold_decision") == "ANSWER"]
    predictions = []
    for row in rows:
        b0 = row.get("baseline_pretrained_decision", "ANSWER")
        prop = row.get("proposed_decision", "")
        unsupported_case = row.get("gold_decision") in {"TICKET", "REJECT"}
        predictions.append(
            {
                "query_id": row.get("query_id"),
                "query": row.get("query"),
                "gold_decision": row.get("gold_decision"),
                "baseline_decision": b0,
                "proposed_decision": prop,
                "unsupported_case": unsupported_case,
                "baseline_unsupported_answer": unsupported_case and b0 == "ANSWER",
                "proposed_unsupported_answer": unsupported_case and prop == "ANSWER",
                "prevented_by_proposed": unsupported_case and b0 == "ANSWER" and prop in {"TICKET", "REJECT"},
            }
        )
    baseline_unsupported_count = sum(p["baseline_unsupported_answer"] for p in predictions)
    proposed_unsupported_count = sum(p["proposed_unsupported_answer"] for p in predictions)
    prevented = sum(p["prevented_by_proposed"] for p in predictions)
    metrics = {
        "unsupported_case_count": len(unsupported),
        "answerable_case_count": len(answerable),
        "baseline": {
            "UnsupportedAnswerRate": baseline_unsupported_count / max(1, len(unsupported)),
            "UnsupportedAnswerCount": baseline_unsupported_count,
            "SafeActionRate": 0.0,
            "OODAnswerRate": _rate(rows, "REJECT", "baseline_pretrained_decision", "ANSWER"),
            "TicketMissRate": _rate(rows, "TICKET", "baseline_pretrained_decision", "ANSWER"),
            "FalseRejectOnAnswerableRate": _rate(rows, "ANSWER", "baseline_pretrained_decision", "REJECT"),
        },
        "proposed": {
            "UnsupportedAnswerRate": proposed_unsupported_count / max(1, len(unsupported)),
            "UnsupportedAnswerCount": proposed_unsupported_count,
            "UnsupportedAnswerPreventionCount": prevented,
            "UnsupportedAnswerPreventionRate": prevented / max(1, baseline_unsupported_count),
            "SafeActionRate": sum(
                row.get("gold_decision") in {"TICKET", "REJECT"} and row.get("proposed_decision") in {"TICKET", "REJECT"}
                for row in rows
            )
            / max(1, len(unsupported)),
            "OODAnswerRate": _rate(rows, "REJECT", "proposed_decision", "ANSWER"),
            "TicketMissRate": _rate(rows, "TICKET", "proposed_decision", "ANSWER"),
            "FalseRejectOnAnswerableRate": _rate(rows, "ANSWER", "proposed_decision", "REJECT"),
        },
    }
    return metrics, predictions


def _rate(rows: list[dict], gold: str, pred_field: str, pred_value: str) -> float:
    subset = [r for r in rows if r.get("gold_decision") == gold]
    return sum(r.get(pred_field) == pred_value for r in subset) / max(1, len(subset))


def _write_summary(metrics: dict) -> None:
    b = metrics["baseline"]
    p = metrics["proposed"]
    lines = [
        "# Unsupported Answer Safety",
        "",
        "Since Baseline is a simple RAG system without ticket or reject tools, direct triage-F1 comparison can be misleading. We therefore report unsupported-answer safety: how often a system gives a direct answer when the KB does not contain sufficient evidence, and how many such failures the proposed system prevents.",
        "",
        f"Unsupported cases: `{metrics['unsupported_case_count']}`",
        f"Answerable cases: `{metrics['answerable_case_count']}`",
        "",
        "| Metric | Baseline | Proposed | Delta |",
        "|---|---:|---:|---:|",
    ]
    keys = [
        "UnsupportedAnswerRate",
        "UnsupportedAnswerCount",
        "SafeActionRate",
        "OODAnswerRate",
        "TicketMissRate",
        "FalseRejectOnAnswerableRate",
    ]
    for key in keys:
        lines.append(f"| {key} | {_fmt(b.get(key))} | {_fmt(p.get(key))} | {_fmt_delta(p.get(key), b.get(key))} |")
    lines.extend(
        [
            f"| UnsupportedAnswerPreventionCount | - | {_fmt(p['UnsupportedAnswerPreventionCount'])} | - |",
            f"| UnsupportedAnswerPreventionRate | - | {_fmt(p['UnsupportedAnswerPreventionRate'])} | - |",
        ]
    )
    project_path("outputs", "reports", "unsupported_answer_safety_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_delta(value, base) -> str:
    if isinstance(value, (int, float)) and isinstance(base, (int, float)):
        return f"{value - base:+.4f}"
    return "-"


if __name__ == "__main__":
    main()
