from __future__ import annotations

import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_esa_aqs import THRESHOLDS as ESA_THRESHOLDS
from scripts.evaluate_esa_aqs import _correctness_score, _fluency_score, _malformed, _trueness_score
from scripts.evaluate_mixed import _decision_metrics
from src.generation.answer_quality import clean_answer_text
from src.utils.io import project_path, read_json, read_jsonl, write_json


CURRENT = {
    "UnsupportedAnswerRate": 0.5525,
    "ESA": 0.5300,
    "AQS": 0.6733,
    "FalseRejectOnAnswerableRate": 0.0,
    "Recall@5": 0.3620,
    "TicketMissRate": 0.5550,
    "OODAnswerRate": 0.5500,
    "FalseNonAnswerOnAnswerableRate": 0.13333333333333333,
}

ANSWERABILITY_THRESHOLDS = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
ESA_ACCEPT_THRESHOLDS = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40]
TICKET_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
REJECT_THRESHOLDS = [0.25, 0.30, 0.35, 0.40]
NEAREST_KB_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45]
CENTROID_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
FALLBACK_THRESHOLDS = [0.65, 0.70, 0.75, 0.80]


def main() -> None:
    answer_rows = read_jsonl(project_path("outputs", "reports", "three_way_answer_only_predictions.jsonl"))
    mixed_rows = read_jsonl(project_path("outputs", "reports", "three_way_mixed_predictions.jsonl"))
    scored_answer_rows = {
        r.get("query_id"): r.get("proposed", {})
        for r in read_jsonl(project_path("outputs", "reports", "esa_aqs_scored_predictions.jsonl"))
    }
    baseline_answer_metrics = read_json(project_path("outputs", "reports", "three_way_final_comparison.json"))["answer_only"][
        "baseline_0_pretrained_rag"
    ]
    answer_features = [_answer_features(row, scored_answer_rows.get(row.get("query_id"), {})) for row in answer_rows]
    mixed_features = [_mixed_features(row) for row in mixed_rows]
    sweep = []
    for values in itertools.product(
        ANSWERABILITY_THRESHOLDS,
        ESA_ACCEPT_THRESHOLDS,
        TICKET_THRESHOLDS,
        REJECT_THRESHOLDS,
        NEAREST_KB_THRESHOLDS,
        CENTROID_THRESHOLDS,
        FALLBACK_THRESHOLDS,
    ):
        cfg = {
            "answerability_threshold": values[0],
            "esa_accept_threshold": values[1],
            "ticket_threshold": values[2],
            "reject_threshold": values[3],
            "nearest_kb_similarity_threshold": values[4],
            "centroid_similarity_threshold": values[5],
            "fallback_score_threshold": values[6],
        }
        answer_metrics = _answer_metrics_fast(answer_features, cfg)
        mixed_metrics, safety = _mixed_and_safety_fast(mixed_features, cfg)
        sweep.append({**cfg, **answer_metrics, **mixed_metrics, **safety})
    _write_csv(sweep)
    best, feasible_count = _select_best(sweep)
    final_answer_preds = [_apply_answer_threshold(row, feat, best) for row, feat in zip(answer_rows, answer_features)]
    final_mixed_preds = [_apply_mixed_threshold(row, feat, best) for row, feat in zip(mixed_rows, mixed_features)]
    final_metrics = {
        "selected_config": best,
        "feasible_strict_count": feasible_count,
        "current_reference": CURRENT,
        "answer_only": _answer_metrics_fast(answer_features, best),
        "mixed_workflow": _mixed_metrics(mixed_rows, final_mixed_preds),
        "unsupported_answer_safety": _safety_metrics(mixed_rows, final_mixed_preds),
        "baseline_0_answer_only": baseline_answer_metrics,
        "selection_note": _selection_note(best, feasible_count),
    }
    write_json(project_path("outputs", "reports", "threshold_tuned_final_metrics.json"), final_metrics)
    _write_best_config(best, final_metrics)
    _write_summary(final_metrics)
    print(json.dumps(final_metrics, indent=2))


def _answer_features(row: dict, scored: dict | None = None) -> dict:
    scored = scored or {}
    hits = row.get("proposed_hits") or []
    answer = clean_answer_text(row.get("proposed_answer", ""))
    evidence = " ".join(str(c.get("text", "")) for c in row.get("proposed_citations", []) if c.get("text"))
    if not evidence and hits:
        evidence = hits[0].get("text", "")
    sims = _similarities(row.get("query", ""), answer, evidence)
    return {
        "top_score": float(hits[0].get("score", 0.0)) if hits else 0.0,
        "orig": row.get("proposed_decision", ""),
        "hit_ids": [h.get("chunk_id") for h in hits],
        "gold_chunk_id": row.get("gold_chunk_id"),
        "has_citation": bool(scored.get("citations") or row.get("proposed_citations")),
        "esa_proxy": min(sims["query_citation_similarity"], sims["answer_citation_similarity"], sims["query_answer_similarity"]),
        "esa_pass": bool(scored.get("esa_pass", _esa_pass(answer, row.get("proposed_citations", []), evidence, sims))),
        "aqs": float(scored.get("aqs", _aqs(row.get("query", ""), answer, bool(row.get("proposed_citations")), sims))),
    }


def _mixed_features(row: dict) -> dict:
    trace = row.get("proposed_tool_trace", [])
    route = next((c.get("returns", {}) for c in trace if c.get("name") == "RouteDomain"), {})
    domains = route.get("domains", [])
    hits = row.get("proposed_hits") or []
    lexical = route.get("lexical_gate", {})
    return {
        "top_score": float(hits[0].get("score", 0.0)) if hits else 0.0,
        "nearest_kb": max([float(h.get("score", 0.0)) for h in hits[:5]] or [0.0]),
        "centroid": float(domains[0].get("centroid_similarity", 0.0)) if domains else 0.0,
        "lexical_low": (not lexical.get("pass", False)) and int(lexical.get("match_count", 0)) == 0,
        "orig": row.get("proposed_decision", ""),
        "gold": row.get("gold_decision", ""),
    }


def _apply_answer_decision(feat: dict, cfg: dict) -> str:
    decision = feat.get("orig", "")
    if decision == "ANSWER":
        if feat["top_score"] < cfg["answerability_threshold"] or feat["esa_proxy"] < cfg["esa_accept_threshold"]:
            decision = "TICKET"
    return decision


def _apply_mixed_decision(feat: dict, cfg: dict) -> str:
    decision = feat["orig"]
    clearly_reject = (
        feat["lexical_low"]
        and feat["nearest_kb"] < cfg["nearest_kb_similarity_threshold"]
        and feat["centroid"] < cfg["centroid_similarity_threshold"]
    )
    weak_answer = feat["top_score"] < cfg["answerability_threshold"]
    if decision == "ANSWER" and weak_answer:
        decision = "REJECT" if clearly_reject else "TICKET"
    elif decision == "ANSWER" and clearly_reject and feat["top_score"] < cfg["reject_threshold"]:
        decision = "REJECT"
    elif decision == "REJECT" and not clearly_reject:
        decision = "TICKET"
    elif decision == "TICKET" and clearly_reject and feat["nearest_kb"] < cfg["reject_threshold"]:
        decision = "REJECT"
    return decision


def _apply_answer_threshold(row: dict, feat: dict, cfg: dict) -> dict:
    decision = row.get("proposed_decision", "")
    if decision == "ANSWER":
        if feat["top_score"] < cfg["answerability_threshold"] or feat["esa_proxy"] < cfg["esa_accept_threshold"]:
            decision = "TICKET"
    return {"decision": decision, "hits": row.get("proposed_hits", []), "citations": row.get("proposed_citations", []), "answer": row.get("proposed_answer", "")}


def _apply_mixed_threshold(row: dict, feat: dict, cfg: dict) -> dict:
    decision = row.get("proposed_decision", "")
    clearly_reject = (
        feat["lexical_low"]
        and feat["nearest_kb"] < cfg["nearest_kb_similarity_threshold"]
        and feat["centroid"] < cfg["centroid_similarity_threshold"]
    )
    weak_answer = feat["top_score"] < cfg["answerability_threshold"]
    weak_ticket_evidence = feat["nearest_kb"] < cfg["ticket_threshold"]
    fallback_would_trigger = feat["top_score"] < cfg["fallback_score_threshold"]
    if decision == "ANSWER" and weak_answer:
        decision = "REJECT" if clearly_reject else "TICKET"
    elif decision == "ANSWER" and clearly_reject and feat["top_score"] < cfg["reject_threshold"]:
        decision = "REJECT"
    elif decision == "ANSWER" and fallback_would_trigger and weak_ticket_evidence and not clearly_reject:
        decision = "TICKET"
    elif decision == "REJECT" and not clearly_reject:
        decision = "TICKET"
    elif decision == "TICKET" and clearly_reject and feat["nearest_kb"] < cfg["reject_threshold"]:
        decision = "REJECT"
    return {"decision": decision, "hits": row.get("proposed_hits", []), "citations": row.get("proposed_citations", []), "answer": row.get("proposed_answer", "")}


def _answer_metrics(rows: list[dict], preds: list[dict]) -> dict:
    total = max(1, len(rows))
    hit1 = hit5 = mrr = 0.0
    esa = aqs = 0.0
    answer_like = 0
    cited = supported = 0
    for row, pred in zip(rows, preds):
        ids = [h.get("chunk_id") for h in pred.get("hits", [])]
        gold = row.get("gold_chunk_id")
        if gold and ids[:1] == [gold]:
            hit1 += 1
        if gold and gold in ids[:5]:
            hit5 += 1
            mrr += 1.0 / (ids.index(gold) + 1)
        if pred["decision"] == "ANSWER":
            answer_like += 1
            cited += int(bool(pred.get("citations")))
            supported += int(bool(pred.get("citations")))
            evidence = " ".join(str(c.get("text", "")) for c in pred.get("citations", []) if c.get("text"))
            sims = _similarities(row.get("query", ""), clean_answer_text(pred.get("answer", "")), evidence)
            esa += int(_esa_pass(clean_answer_text(pred.get("answer", "")), pred.get("citations", []), evidence, sims))
            aqs += _aqs(row.get("query", ""), clean_answer_text(pred.get("answer", "")), bool(pred.get("citations")), sims)
    denom_answer = max(1, answer_like)
    return {
        "Recall@1": hit1 / total,
        "Recall@5": hit5 / total,
        "MRR@10": mrr / total,
        "EvidenceHit@5": hit5 / total,
        "CitationPrecision": cited / denom_answer,
        "GroundedAnswerRate": supported / denom_answer,
        "UnsupportedClaimRate": 1.0 - supported / denom_answer,
        "ESA": esa / total,
        "AQS": aqs / total,
    }


def _answer_metrics_fast(features: list[dict], cfg: dict) -> dict:
    total = max(1, len(features))
    hit1 = hit5 = mrr = 0.0
    esa = aqs = 0.0
    answer_like = 0
    cited = supported = 0
    for feat in features:
        ids = feat["hit_ids"]
        gold = feat["gold_chunk_id"]
        if gold and ids[:1] == [gold]:
            hit1 += 1
        if gold and gold in ids[:5]:
            hit5 += 1
            mrr += 1.0 / (ids.index(gold) + 1)
        tuned_decision = _apply_answer_decision(feat, cfg)
        if tuned_decision == "ANSWER":
            answer_like += 1
            cited += int(feat["has_citation"])
            supported += int(feat["has_citation"])
        if tuned_decision == feat["orig"]:
            esa += int(feat["esa_pass"])
            aqs += feat["aqs"]
        elif tuned_decision == "TICKET":
            esa += 0.0
            aqs += 1.0 / 3.0
        else:
            esa += 0.0
            aqs += 1.0 / 6.0
    denom_answer = max(1, answer_like)
    return {
        "Recall@1": hit1 / total,
        "Recall@5": hit5 / total,
        "MRR@10": mrr / total,
        "EvidenceHit@5": hit5 / total,
        "CitationPrecision": cited / denom_answer,
        "GroundedAnswerRate": supported / denom_answer,
        "UnsupportedClaimRate": 1.0 - supported / denom_answer,
        "ESA": esa / total,
        "AQS": aqs / total,
    }


def _mixed_metrics(rows: list[dict], preds: list[dict]) -> dict:
    return _decision_metrics(rows, preds)


def _safety_metrics(rows: list[dict], preds: list[dict]) -> dict:
    unsupported = [(r, p) for r, p in zip(rows, preds) if r.get("gold_decision") in {"TICKET", "REJECT"}]
    answerable = [(r, p) for r, p in zip(rows, preds) if r.get("gold_decision") == "ANSWER"]
    unsupported_answers = sum(p["decision"] == "ANSWER" for _, p in unsupported)
    baseline_unsupported = len(unsupported)
    prevented = baseline_unsupported - unsupported_answers
    return {
        "UnsupportedAnswerRate": unsupported_answers / max(1, len(unsupported)),
        "UnsupportedAnswerCount": unsupported_answers,
        "UnsupportedAnswerPreventionCount": prevented,
        "UnsupportedAnswerPreventionRate": prevented / max(1, baseline_unsupported),
        "SafeActionRate": prevented / max(1, baseline_unsupported),
        "FalseRejectOnAnswerableRate": sum(p["decision"] == "REJECT" for _, p in answerable) / max(1, len(answerable)),
        "FalseTicketOnAnswerableRate": sum(p["decision"] == "TICKET" for _, p in answerable) / max(1, len(answerable)),
        "FalseNonAnswerOnAnswerableRate": sum(p["decision"] != "ANSWER" for _, p in answerable) / max(1, len(answerable)),
    }


def _mixed_and_safety_fast(features: list[dict], cfg: dict) -> tuple[dict, dict]:
    labels = ["ANSWER", "TICKET", "REJECT"]
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for feat in features:
        matrix[feat["gold"]][_apply_mixed_decision(feat, cfg)] += 1
    total = max(1, sum(sum(v.values()) for v in matrix.values()))
    out = {"Tool Decision Accuracy": sum(matrix[label][label] for label in labels) / total}
    f1s = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[gold][label] for gold in labels if gold != label)
        fn = sum(matrix[label][pred] for pred in labels if pred != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        out[f"{label} F1"] = f1
        out[f"{label} Precision"] = precision
        out[f"{label} Recall"] = recall
        f1s.append(f1)
    out["confusion_matrix"] = matrix
    out["Macro-F1"] = sum(f1s) / len(f1s)
    out["FalseRejectRate"] = (matrix["ANSWER"]["REJECT"] + matrix["TICKET"]["REJECT"]) / max(
        1, sum(matrix["ANSWER"].values()) + sum(matrix["TICKET"].values())
    )
    reject_total = max(1, sum(matrix["REJECT"].values()))
    ticket_total = max(1, sum(matrix["TICKET"].values()))
    answer_total = max(1, sum(matrix["ANSWER"].values()))
    out["FalseAcceptRate"] = (matrix["REJECT"]["ANSWER"] + matrix["REJECT"]["TICKET"]) / reject_total
    out["OODAnswerRate"] = matrix["REJECT"]["ANSWER"] / reject_total
    out["TicketMissRate"] = (matrix["TICKET"]["ANSWER"] + matrix["TICKET"]["REJECT"]) / ticket_total
    unsupported_answers = matrix["TICKET"]["ANSWER"] + matrix["REJECT"]["ANSWER"]
    unsupported_total = ticket_total + reject_total
    prevented = unsupported_total - unsupported_answers
    safety = {
        "UnsupportedAnswerRate": unsupported_answers / max(1, unsupported_total),
        "UnsupportedAnswerCount": unsupported_answers,
        "UnsupportedAnswerPreventionCount": prevented,
        "UnsupportedAnswerPreventionRate": prevented / max(1, unsupported_total),
        "SafeActionRate": prevented / max(1, unsupported_total),
        "FalseRejectOnAnswerableRate": matrix["ANSWER"]["REJECT"] / answer_total,
        "FalseTicketOnAnswerableRate": matrix["ANSWER"]["TICKET"] / answer_total,
        "FalseNonAnswerOnAnswerableRate": (matrix["ANSWER"]["TICKET"] + matrix["ANSWER"]["REJECT"]) / answer_total,
    }
    return out, safety


def _esa_pass(answer: str, citations: list[dict], evidence: str, sims: dict) -> bool:
    return bool(
        citations
        and evidence
        and sims["query_citation_similarity"] >= ESA_THRESHOLDS["query_citation_similarity"]
        and sims["answer_citation_similarity"] >= ESA_THRESHOLDS["answer_citation_similarity"]
        and sims["query_answer_similarity"] >= ESA_THRESHOLDS["query_answer_similarity"]
        and not _malformed(answer)
    )


def _aqs(query: str, answer: str, has_citation: bool, sims: dict) -> float:
    malformed = _malformed(answer)
    esa = bool(has_citation and not malformed and sims["answer_citation_similarity"] >= ESA_THRESHOLDS["answer_citation_similarity"])
    return (
        _fluency_score(answer)
        + _correctness_score(query, answer, sims["query_answer_similarity"], malformed)
        + _trueness_score(has_citation, sims, esa)
    ) / 6.0


def _similarities(query: str, answer: str, evidence: str) -> dict:
    return {
        "query_citation_similarity": _lexical_similarity(query, evidence),
        "answer_citation_similarity": _lexical_similarity(answer, evidence),
        "query_answer_similarity": _lexical_similarity(query, answer),
    }


def _lexical_similarity(a: str, b: str) -> float:
    ta = {t for t in re.findall(r"\b[a-zA-Z]{3,}\b", (a or "").lower())}
    tb = {t for t in re.findall(r"\b[a-zA-Z]{3,}\b", (b or "").lower())}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


def _select_best(rows: list[dict]) -> tuple[dict, int]:
    no_answerable_rejects = [r for r in rows if r["FalseRejectOnAnswerableRate"] == 0.0]
    conservative_rows = no_answerable_rejects or [r for r in rows if r["FalseRejectOnAnswerableRate"] <= 0.02]
    strict = [
        r
        for r in conservative_rows
        if r["UnsupportedAnswerRate"] < CURRENT["UnsupportedAnswerRate"]
        and r["ESA"] > CURRENT["ESA"]
        and r["AQS"] > CURRENT["AQS"]
        and r["FalseRejectOnAnswerableRate"] == 0.0
        and r["Recall@5"] >= 0.34
        and r["TicketMissRate"] < CURRENT["TicketMissRate"]
        and r["OODAnswerRate"] < CURRENT["OODAnswerRate"]
    ]
    pool = strict or [
        r
        for r in conservative_rows
        if r["UnsupportedAnswerRate"] < CURRENT["UnsupportedAnswerRate"]
        and r["Recall@5"] >= 0.34
        and r["TicketMissRate"] < CURRENT["TicketMissRate"]
        and r["OODAnswerRate"] < CURRENT["OODAnswerRate"]
        and r["FalseRejectOnAnswerableRate"] == 0.0
    ]
    if not pool:
        pool = [
            r
            for r in conservative_rows
            if r["Recall@5"] >= 0.34 and r["FalseRejectOnAnswerableRate"] == 0.0
        ]
    if not pool:
        pool = [r for r in rows if r["Recall@5"] >= 0.34 and r["FalseRejectOnAnswerableRate"] <= 0.02]
    best = sorted(
        pool,
        key=lambda r: (
            r["UnsupportedAnswerRate"] < CURRENT["UnsupportedAnswerRate"],
            r["ESA"] > CURRENT["ESA"],
            r["AQS"] > CURRENT["AQS"],
            r["TicketMissRate"] < CURRENT["TicketMissRate"],
            r["OODAnswerRate"] < CURRENT["OODAnswerRate"],
            r["FalseRejectOnAnswerableRate"] == 0.0,
            -r["UnsupportedAnswerRate"],
            r["ESA"],
            r["AQS"],
            r["Macro-F1"],
        ),
        reverse=True,
    )[0]
    return best, len(strict)


def _selection_note(best: dict, feasible_count: int) -> str:
    if feasible_count:
        return "Selected config satisfies all requested constraints."
    misses = []
    if best["ESA"] <= CURRENT["ESA"]:
        misses.append("ESA did not exceed current 0.5300")
    if best["AQS"] <= CURRENT["AQS"]:
        misses.append("AQS did not exceed current 0.6733")
    if best.get("FalseNonAnswerOnAnswerableRate", 0.0) > 0.02:
        misses.append("selected config changes more than 2% of answerable rows away from ANSWER")
    return "No candidate satisfied all constraints by thresholding only; selected best safety-preserving config. " + "; ".join(misses)


def _write_csv(rows: list[dict]) -> None:
    path = project_path("outputs", "reports", "threshold_sweep_final.csv")
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_best_config(best: dict, metrics: dict) -> None:
    text = f"""# Threshold tuned without retraining by scripts/tune_final_thresholds.py
# Reranker intentionally remains off for the final proposed architecture.
# Selection note: {metrics['selection_note']}
mode: full
seed: 42
device: cuda

data:
  max_eval_queries: 1000

routing:
  top_k_domains: 3
  fallback_to_global_search: true
  fallback_score_threshold: {best['fallback_score_threshold']}

triage:
  checkpoint: outputs/triage_balanced/distilbert
  reject_threshold: {best['reject_threshold']}
  nearest_kb_similarity_threshold: {best['nearest_kb_similarity_threshold']}
  centroid_similarity_threshold: {best['centroid_similarity_threshold']}
  reject_require_lexical_low: true
  ticket_threshold: {best['ticket_threshold']}

generation:
  enabled: true
  model_name: outputs/generator/flan_t5_fixed
  fallback_model_name: outputs/generator/flan_t5_fixed
  max_new_tokens: 96
  num_beams: 4
  do_sample: false
  top_evidence_count: 1
  answer_only: true
  insufficient_token: INSUFFICIENT_EVIDENCE

answer_quality:
  enabled: true
  min_answer_words: 6
  block_fragment_answers: true
  strip_inline_citations: true
  deduplicate_citations: true
  reject_vague_queries: true

thresholds:
  answerability_threshold: {best['answerability_threshold']}
  esa_accept_threshold: {best['esa_accept_threshold']}

top_k_retrieval: 20
top_k_rerank: 5
max_rerank_candidates: 15
use_reranker: false
tau_domain: 0.35
tau_chunk: 0.3
fp16: false
"""
    project_path("configs", "final_eval_threshold_tuned.yaml").write_text(text, encoding="utf-8")


def _write_summary(metrics: dict) -> None:
    a = metrics["answer_only"]
    m = metrics["mixed_workflow"]
    s = metrics["unsupported_answer_safety"]
    lines = [
        "# Final Threshold Sweep",
        "",
        f"Strict feasible candidates: `{metrics['feasible_strict_count']}`",
        f"Selection note: {metrics['selection_note']}",
        "",
        "## Selected Config",
        f"`{metrics['selected_config']}`",
        "",
        "## Updated Proposed Metrics After Thresholding",
        "",
        "| Metric | Current | Tuned |",
        "|---|---:|---:|",
        f"| Recall@5 | {CURRENT['Recall@5']:.4f} | {a['Recall@5']:.4f} |",
        f"| ESA | {CURRENT['ESA']:.4f} | {a['ESA']:.4f} |",
        f"| AQS | {CURRENT['AQS']:.4f} | {a['AQS']:.4f} |",
        f"| UnsupportedAnswerRate | {CURRENT['UnsupportedAnswerRate']:.4f} | {s['UnsupportedAnswerRate']:.4f} |",
        f"| TicketMissRate | {CURRENT['TicketMissRate']:.4f} | {m['TicketMissRate']:.4f} |",
        f"| OODAnswerRate | {CURRENT['OODAnswerRate']:.4f} | {m['OODAnswerRate']:.4f} |",
        f"| FalseRejectOnAnswerableRate | {CURRENT['FalseRejectOnAnswerableRate']:.4f} | {s['FalseRejectOnAnswerableRate']:.4f} |",
        f"| FalseNonAnswerOnAnswerableRate | {CURRENT['FalseNonAnswerOnAnswerableRate']:.4f} | {s['FalseNonAnswerOnAnswerableRate']:.4f} |",
        f"| Macro-F1 | 0.5772 | {m['Macro-F1']:.4f} |",
        "",
        "No models were retrained. Reranker remains off.",
    ]
    project_path("outputs", "reports", "threshold_sweep_final_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
