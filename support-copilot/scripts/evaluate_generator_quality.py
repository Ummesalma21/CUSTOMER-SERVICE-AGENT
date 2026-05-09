from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.answer_quality import clean_answer_text, is_fragment_answer
from src.generation.extractive_synthesizer import synthesize_extractive_answer
from src.generation.grounded_generator import generate_grounded_answer
from src.utils.io import project_path, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/processed/generator_test.jsonl")
    parser.add_argument("--model", default="outputs/generator/flan_t5_fixed")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--predictions-out", default="outputs/reports/generator_fixed_quality_predictions.jsonl")
    parser.add_argument("--metrics-out", default="outputs/reports/generator_fixed_quality_metrics.json")
    parser.add_argument("--summary-out", default="outputs/reports/generator_fixed_quality_summary.md")
    args = parser.parse_args()
    rows = read_jsonl(project_path(*Path(args.test_file).parts))
    if args.max_rows:
        rows = rows[: args.max_rows]
    predictions = []
    extractive_scores = []
    generator_scores = []
    for row in rows:
        query = row.get("query", "")
        evidence = row.get("evidence", [])[:3]
        reference = row.get("target") or row.get("target_answer") or row.get("reference_answer") or ""
        ext = synthesize_extractive_answer(query, evidence)
        gen = generate_grounded_answer(
            query=query,
            evidence_passages=evidence,
            model_name=args.model,
            fallback_model_name=args.model,
            max_new_tokens=96,
            num_beams=4,
            do_sample=False,
        )
        ext_score = _score(ext.get("answer") or "", reference, query, evidence, status=ext.get("status"), citation_attached=bool(evidence))
        gen_score = _score(gen.get("answer") or "", reference, query, evidence, status=gen.get("status"), citation_attached=bool(evidence))
        extractive_scores.append(ext_score)
        generator_scores.append(gen_score)
        predictions.append(
            {
                "query": query,
                "reference": reference,
                "evidence": evidence,
                "extractive_answer": ext.get("answer"),
                "extractive_status": ext.get("status"),
                "fixed_generator_answer": gen.get("answer"),
                "fixed_generator_status": gen.get("status"),
                "fixed_generator_model": gen.get("model_name"),
                "fixed_generator_fallback_reason": gen.get("fallback_reason"),
                "extractive_scores": ext_score,
                "fixed_generator_scores": gen_score,
            }
        )
    metrics = {
        "test_file": args.test_file,
        "count": len(rows),
        "extractive_fallback": _aggregate(extractive_scores),
        "fixed_generator": _aggregate(generator_scores),
    }
    write_json(project_path(*Path(args.metrics_out).parts), metrics)
    write_jsonl(project_path(*Path(args.predictions_out).parts), predictions)
    project_path(*Path(args.summary_out).parts).write_text(_summary(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _score(answer: str, reference: str, query: str, evidence: list[dict], status: str | None, citation_attached: bool) -> dict:
    cleaned = clean_answer_text(answer)
    words = _tokens(cleaned)
    return {
        "AnswerTokenF1": _token_f1(cleaned, reference),
        "ROUGE-L": _rouge_l(cleaned, reference),
        "NoFragment": not is_fragment_answer(cleaned),
        "Fragment": is_fragment_answer(cleaned),
        "QuestionAsAnswer": cleaned.strip().endswith("?"),
        "QueryCopy": _query_copy(cleaned, query),
        "EmptyOrInvalidAnswer": len(words) < 3,
        "AverageAnswerLengthWords": len(words),
        "EvidenceSupportedHeuristic": _evidence_supported(cleaned, evidence),
        "InsufficientEvidence": status == "insufficient_evidence" or cleaned.strip() == "INSUFFICIENT_EVIDENCE",
        "CitationAttached": citation_attached,
    }


def _aggregate(scores: list[dict]) -> dict:
    n = max(1, len(scores))
    return {
        "AnswerTokenF1": sum(s["AnswerTokenF1"] for s in scores) / n,
        "ROUGE-L": sum(s["ROUGE-L"] for s in scores) / n,
        "NoFragmentRate": sum(s["NoFragment"] for s in scores) / n,
        "FragmentRate": sum(s["Fragment"] for s in scores) / n,
        "QuestionAsAnswerRate": sum(s["QuestionAsAnswer"] for s in scores) / n,
        "QueryCopyRate": sum(s["QueryCopy"] for s in scores) / n,
        "EmptyOrInvalidAnswerRate": sum(s["EmptyOrInvalidAnswer"] for s in scores) / n,
        "AverageAnswerLengthWords": sum(s["AverageAnswerLengthWords"] for s in scores) / n,
        "EvidenceSupportedHeuristicRate": sum(s["EvidenceSupportedHeuristic"] for s in scores) / n,
        "INSUFFICIENT_EVIDENCE_rate": sum(s["InsufficientEvidence"] for s in scores) / n,
        "CitationAttachedRate": sum(s["CitationAttached"] for s in scores) / n,
    }


def _token_f1(predicted: str, reference: str) -> float:
    pred = _tokens(predicted)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    common = 0
    ref_counts = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    for tok in pred:
        if ref_counts.get(tok, 0):
            common += 1
            ref_counts[tok] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _rouge_l(predicted: str, reference: str) -> float:
    pred = _tokens(predicted)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for token in pred:
        curr = [0]
        for j, ref_token in enumerate(ref, start=1):
            curr.append(prev[j - 1] + 1 if token == ref_token else max(prev[j], curr[-1]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _tokens(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation)
    return [tok for tok in (text or "").lower().translate(table).split() if tok]


def _query_copy(answer: str, query: str) -> bool:
    a = set(_tokens(answer))
    q = set(_tokens(query))
    return bool(a and q and len(a & q) / max(1, len(a)) > 0.85)


def _evidence_supported(answer: str, evidence: list[dict]) -> bool:
    a = {tok for tok in _tokens(answer) if len(tok) >= 4}
    e = {tok for item in evidence for tok in _tokens(item.get("text", "")) if len(tok) >= 4}
    return bool(a and e and len(a & e) / max(1, len(a)) >= 0.35)


def _summary(metrics: dict) -> str:
    lines = [
        "# Fixed Generator Quality",
        "",
        f"Test file: `{metrics['test_file']}`",
        f"Rows: `{metrics['count']}`",
        "",
        "| Metric | Extractive fallback | Fixed FLAN-T5 |",
        "|---|---:|---:|",
    ]
    keys = sorted(metrics["fixed_generator"].keys())
    for key in keys:
        lines.append(f"| {key} | {metrics['extractive_fallback'][key]} | {metrics['fixed_generator'][key]} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
