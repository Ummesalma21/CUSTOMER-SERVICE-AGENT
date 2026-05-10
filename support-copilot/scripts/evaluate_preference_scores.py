from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preference.score_candidates import score_answer
from src.utils.io import project_path, read_json, read_jsonl, write_json


SYSTEM_FIELDS = {
    "baseline": "baseline_pretrained_answer",
    "baseline_1_finetuned_rag": "baseline_1_finetuned_answer",
    "proposed": "proposed_answer",
}


def main() -> None:
    answer_rows = read_jsonl(project_path("outputs", "reports", "three_way_answer_only_predictions.jsonl"))
    mixed_rows = read_jsonl(project_path("outputs", "reports", "three_way_mixed_predictions.jsonl"))
    preference_train = read_json(project_path("outputs", "preference", "metrics.json"), {})
    metrics = {
        "preference_model": preference_train,
        "rubric": read_json(project_path("outputs", "preference", "model.json"), {}),
        "answer_only": _score_rows(answer_rows, gold_field="gold_triage"),
        "mixed_workflow": _score_rows(mixed_rows, gold_field="gold_decision"),
    }
    write_json(project_path("outputs", "reports", "preference_score_comparison.json"), metrics)
    _write_summary(metrics)
    print(json.dumps(metrics, indent=2))


def _score_rows(rows: list[dict], gold_field: str) -> dict:
    out = {}
    for system, answer_field in SYSTEM_FIELDS.items():
        scores = []
        wins_vs_baseline = 0
        comparable = 0
        for row in rows:
            expected = row.get(gold_field) or "ANSWER"
            score = score_answer(_answer_for_score(row, answer_field), expected)
            scores.append(score)
            if system != "baseline":
                b0 = score_answer(_answer_for_score(row, SYSTEM_FIELDS["baseline"]), expected)
                wins_vs_baseline += int(score > b0)
                comparable += 1
        out[system] = {
            "mean_preference_score": sum(scores) / max(1, len(scores)),
            "min_preference_score": min(scores) if scores else 0.0,
            "max_preference_score": max(scores) if scores else 0.0,
            "rows": len(scores),
        }
        if system != "baseline":
            out[system]["win_rate_vs_baseline"] = wins_vs_baseline / max(1, comparable)
    return out


def _answer_for_score(row: dict, answer_field: str) -> str:
    answer = row.get(answer_field, "") or ""
    if "[doc_id=" in answer:
        return answer
    prefix = answer_field.replace("_answer", "")
    decision = row.get(f"{prefix}_decision", "")
    if decision != "ANSWER":
        return answer
    citations = row.get(f"{prefix}_citations") or row.get(f"{prefix}_hits") or []
    if not citations:
        return answer
    c = citations[0]
    if not c.get("doc_id") or not c.get("chunk_id"):
        return answer
    span = c.get("span") or f"{c.get('span_start', '')}-{c.get('span_end', '')}"
    return f"{answer} [doc_id={c.get('doc_id')}, chunk_id={c.get('chunk_id')}, span={span}]"


def _write_summary(metrics: dict) -> None:
    lines = [
        "# Preference/Rubric Score Comparison",
        "",
        "Rubric source: `src/preference/score_candidates.py`",
        f"Preference training metrics: `{metrics['preference_model']}`",
        f"Rubric artifact: `{metrics['rubric']}`",
        "",
        "The score is a lightweight rubric score, not full preference optimization. It rewards cited answers, insufficient-evidence ticket style, out-of-domain rejection style, concise outputs, and an extra cited-answer bonus for ANSWER examples.",
        "",
        "## Answer-Only",
        "",
        "| System | Mean Preference Score | Win Rate vs Baseline | Rows |",
        "|---|---:|---:|---:|",
    ]
    for label, key in [
        ("Baseline", "baseline"),
        ("Baseline-1 Fine-tuned RAG", "baseline_1_finetuned_rag"),
        ("Proposed", "proposed"),
    ]:
        row = metrics["answer_only"][key]
        win = row.get("win_rate_vs_baseline")
        win_text = "-" if win is None else f"{win:.4f}"
        lines.append(f"| {label} | {row['mean_preference_score']:.4f} | {win_text} | {row['rows']} |")
    lines.extend(
        [
            "",
            "## Mixed Workflow",
            "",
            "| System | Mean Preference Score | Win Rate vs Baseline | Rows |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in [
        ("Baseline", "baseline"),
        ("Baseline-1 Fine-tuned RAG", "baseline_1_finetuned_rag"),
        ("Proposed", "proposed"),
    ]:
        row = metrics["mixed_workflow"][key]
        win = row.get("win_rate_vs_baseline")
        win_text = "-" if win is None else f"{win:.4f}"
        lines.append(f"| {label} | {row['mean_preference_score']:.4f} | {win_text} | {row['rows']} |")
    project_path("outputs", "reports", "preference_score_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
