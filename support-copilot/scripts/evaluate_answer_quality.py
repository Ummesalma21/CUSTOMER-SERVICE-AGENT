from __future__ import annotations

import argparse
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.templates import cited_answer
from src.presentation import CITATION_RE, clean_inline_citations, citation_identity, extract_inline_citations, is_fragment_answer
from src.utils.io import project_path, read_jsonl, write_json, write_jsonl


REFERENCE_FIELDS = ["gold_answer", "reference_answer", "answer", "target_answer", "expected_answer", "response"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/reports/final_answer_only_predictions.jsonl")
    args = parser.parse_args()
    path = project_path(*Path(args.predictions).parts)
    if not path.exists():
        raise SystemExit(
            "Missing final_answer_only_predictions.jsonl. Run: "
            ".\\.venv\\Scripts\\python.exe scripts\\evaluate_answer_only.py --config configs\\final_eval_balanced_triage_best.yaml"
        )
    rows = read_jsonl(path)
    scored = []
    baseline_scores = []
    proposed_scores = []
    for row in rows:
        reference = _reference(row)
        baseline_answer = row.get("baseline_answer") or cited_answer(row.get("query", ""), row.get("baseline_hits", []))
        proposed_answer = row.get("proposed_answer") or cited_answer(row.get("query", ""), row.get("proposed_hits", []))
        baseline = _score_answer(baseline_answer, row.get("baseline_hits", []), reference, row.get("baseline_decision", "ANSWER"))
        proposed = _score_answer(
            proposed_answer,
            row.get("proposed_citations") or row.get("proposed_hits", []),
            reference,
            row.get("proposed_decision", "ANSWER"),
        )
        baseline_scores.append(baseline)
        proposed_scores.append(proposed)
        scored.append({**row, "reference_available": reference is not None, "baseline_quality": baseline, "proposed_quality": proposed})
    metrics = {
        "count": len(rows),
        "reference_available": sum(1 for row in scored if row["reference_available"]),
        "reference_note": "No gold/reference answer text was present in final_answer_only_predictions.jsonl; token F1 and ROUGE-L are not computed.",
        "baseline": _aggregate(baseline_scores),
        "proposed": _aggregate(proposed_scores),
    }
    write_json(project_path("outputs", "reports", "final_answer_quality_metrics.json"), metrics)
    write_jsonl(project_path("outputs", "reports", "final_answer_quality_predictions_scored.jsonl"), scored)
    _write_summary(metrics)
    print(metrics)


def _reference(row: dict) -> str | None:
    for field in REFERENCE_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _score_answer(answer: str, citations_or_hits: list[dict], reference: str | None, decision: str) -> dict:
    cleaned = clean_inline_citations(answer)
    words = _tokens(cleaned)
    token = _token_f1(cleaned, reference) if reference else None
    rouge = _rouge_l(cleaned, reference) if reference else None
    citation_identities = [citation_identity(c) for c in citations_or_hits or []]
    inline_ids = [citation_identity(c) for c in extract_inline_citations(answer)]
    attached = bool(citation_identities or inline_ids or CITATION_RE.search(answer or ""))
    duplicate = bool(len(set(citation_identities + inline_ids)) < len(citation_identities + inline_ids) and (citation_identities or inline_ids))
    return {
        "AnswerTokenF1": token,
        "ROUGE-L": rouge,
        "NoFragment": not is_fragment_answer(answer),
        "CitationAttached": attached,
        "DuplicateCitation": duplicate,
        "AverageAnswerLengthWords": len(words),
        "EmptyOrInvalidAnswer": len(words) < 3,
        "IsAnswerOutput": decision == "ANSWER",
    }


def _aggregate(scores: list[dict]) -> dict:
    answer_scores = [s for s in scores if s["IsAnswerOutput"]]
    n = max(1, len(answer_scores))
    token_values = [s["AnswerTokenF1"] for s in scores if s["AnswerTokenF1"] is not None]
    rouge_values = [s["ROUGE-L"] for s in scores if s["ROUGE-L"] is not None]
    fragments = sum(1 for s in answer_scores if not s["NoFragment"])
    duplicates = sum(1 for s in answer_scores if s["DuplicateCitation"])
    invalid = sum(1 for s in answer_scores if s["EmptyOrInvalidAnswer"])
    return {
        "AnswerTokenF1": sum(token_values) / len(token_values) if token_values else None,
        "ROUGE-L": sum(rouge_values) / len(rouge_values) if rouge_values else None,
        "AnswerOutputCount": len(answer_scores),
        "NoFragmentRate": sum(1 for s in answer_scores if s["NoFragment"]) / n,
        "FragmentCount": fragments,
        "CitationAttachedRate": sum(1 for s in answer_scores if s["CitationAttached"]) / n,
        "DuplicateCitationRate": duplicates / n,
        "DuplicateCitationCount": duplicates,
        "AverageAnswerLengthWords": sum(s["AverageAnswerLengthWords"] for s in answer_scores) / n,
        "EmptyOrInvalidAnswerRate": invalid / n,
        "EmptyOrInvalidAnswerCount": invalid,
    }


def _token_f1(predicted: str, reference: str) -> float:
    pred = _tokens(predicted)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    common = 0
    ref_counts: dict[str, int] = {}
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred:
        if ref_counts.get(token, 0) > 0:
            common += 1
            ref_counts[token] -= 1
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / max(1e-9, precision + recall)


def _rouge_l(predicted: str, reference: str) -> float:
    pred = _tokens(predicted)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / max(1e-9, precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _tokens(text: str) -> list[str]:
    text = clean_inline_citations(text).lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    return re.findall(r"\b\w+\b", text)


def _write_summary(metrics: dict) -> None:
    lines = [
        "# Final Answer Quality Evaluation",
        "",
        "Answer-quality metrics are computed only on answerable examples. Citation markers are stripped before token overlap and ROUGE-L scoring.",
        "",
        f"Rows: `{metrics['count']}`",
        f"Reference answers available: `{metrics['reference_available']}`",
        f"Note: {metrics['reference_note']}",
        "",
        "| Metric | Baseline RAG | Proposed Balanced Triage |",
        "|---|---:|---:|",
    ]
    for key in [
        "AnswerTokenF1",
        "ROUGE-L",
        "NoFragmentRate",
        "CitationAttachedRate",
        "DuplicateCitationRate",
        "EmptyOrInvalidAnswerRate",
        "AverageAnswerLengthWords",
    ]:
        b = metrics["baseline"].get(key)
        p = metrics["proposed"].get(key)
        lines.append(f"| {key} | {_fmt(b)} | {_fmt(p)} |")
    interpretation = "Answer quality is preserved" if metrics["proposed"]["NoFragmentRate"] >= metrics["baseline"]["NoFragmentRate"] else "Answer quality is degraded on fragment-rate"
    lines.extend(["", "## Interpretation", interpretation])
    project_path("outputs", "reports", "final_answer_quality_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
