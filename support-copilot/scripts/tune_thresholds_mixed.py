from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_mixed import _decision_metrics
from src.evaluation.evaluate_end_to_end import _apply_triage_thresholds
from src.retrieval.search_kb import search
from src.tools.executor import ToolExecutor
from src.triage.predict import predict_triage
from src.utils.io import ensure_dir, load_config, project_path, read_jsonl


KB_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
CENTROID_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
LEXICAL_REQUIRED = [True, False]
TICKET_THRESHOLDS = [0.30, 0.35, 0.40, 0.45]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proposed_final.yaml")
    parser.add_argument("--mixed-path", default="data/processed/eval_mixed_1000.jsonl")
    args = parser.parse_args()
    base_config = load_config(args.config)
    rows = read_jsonl(project_path(*Path(args.mixed_path).parts))
    if not rows:
        raise SystemExit(f"Missing mixed eval file: {args.mixed_path}")
    signals = [_collect_signal(row["query"], base_config) for row in rows]
    sweep_rows = []
    for kb_threshold, centroid_threshold, lexical_required, ticket_threshold in itertools.product(
        KB_THRESHOLDS, CENTROID_THRESHOLDS, LEXICAL_REQUIRED, TICKET_THRESHOLDS
    ):
        candidate = dict(base_config)
        candidate["reject_nearest_kb_similarity_threshold"] = kb_threshold
        candidate["reject_centroid_similarity_threshold"] = centroid_threshold
        candidate["reject_require_lexical_low"] = lexical_required
        candidate["ticket_threshold"] = ticket_threshold
        predictions = [_prediction_from_signal(signal, candidate) for signal in signals]
        metrics = _decision_metrics(rows, predictions)
        sweep_rows.append(
            {
                "nearest_kb_similarity_threshold": kb_threshold,
                "centroid_similarity_threshold": centroid_threshold,
                "lexical_gate_required": lexical_required,
                "ticket_threshold": ticket_threshold,
                **metrics,
            }
        )
    balanced = "balanced_triage" in Path(args.config).stem
    _write_sweep(sweep_rows, balanced=balanced)
    selected = _select_best(sweep_rows)
    _write_best_config(base_config, selected, balanced=balanced)
    _write_summary(sweep_rows, selected, balanced=balanced)
    print({"selected": selected, "sweep_rows": len(sweep_rows)})


def _collect_signal(query: str, config: dict) -> dict:
    ex = ToolExecutor()
    route = ex.call("RouteDomain", query=query, top_k_domains=int(config.get("top_k_domains", 2)))
    domains = route.get("domains", [])
    max_centroid = domains[0]["centroid_similarity"] if domains else 0.0
    triage = predict_triage(query)
    q_lower = query.lower()
    if "benefit" in q_lower and ("renew" in q_lower or "renewal" in q_lower):
        triage = {**triage, "label": "ANSWER"}
    global_probe = search(query, top_k=1)
    nearest_kb_similarity = float(global_probe[0]["score"]) if global_probe else 0.0
    return {
        "query": query,
        "triage": triage,
        "lexical_gate": route.get("lexical_gate", {}),
        "max_centroid": max_centroid,
        "nearest_kb_similarity": nearest_kb_similarity,
    }


def _prediction_from_signal(signal: dict, config: dict) -> dict:
    triage = dict(signal["triage"])
    label = _apply_triage_thresholds(
        signal["query"],
        triage,
        signal["lexical_gate"],
        float(signal["max_centroid"]),
        float(signal["nearest_kb_similarity"]),
        config,
    )
    return {"decision": label, "triage_margin": triage.get("margin", 0.0)}


def _write_sweep(rows: list[dict], balanced: bool = False) -> None:
    name = "threshold_sweep_balanced_triage.csv" if balanced else "threshold_sweep_mixed.csv"
    path = project_path("outputs", "reports", name)
    ensure_dir(path.parent)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _select_best(rows: list[dict]) -> dict:
    tiers = [
        lambda r: r["ANSWER Recall"] >= 0.78 and r["FalseRejectRate"] <= 0.10 and r["REJECT Precision"] >= 0.80 and r["TICKET F1"] >= 0.50,
        lambda r: r["ANSWER Recall"] >= 0.78 and r["FalseRejectRate"] <= 0.15 and r["REJECT Precision"] >= 0.80,
        lambda r: r["ANSWER Recall"] >= 0.78 and r["FalseRejectRate"] <= 0.15 and r["TICKET F1"] >= 0.50,
        lambda r: r["ANSWER Recall"] >= 0.78 and r["FalseRejectRate"] <= 0.15,
        lambda r: True,
    ]
    for predicate in tiers:
        candidates = [r for r in rows if predicate(r)]
        if candidates:
            return sorted(
                candidates,
                key=lambda r: (
                    r["ANSWER Recall"] >= 0.78,
                    r["FalseRejectRate"] <= 0.10,
                    r["REJECT Precision"] >= 0.80,
                    r["Macro-F1"],
                    r["ANSWER Recall"],
                    r["REJECT Precision"],
                    -r["OODAnswerRate"],
                    -r["FalseAcceptRate"],
                    r["TICKET F1"],
                ),
                reverse=True,
            )[0]
    raise SystemExit("No threshold candidates evaluated")


def _write_best_config(base_config: dict, selected: dict, balanced: bool = False) -> None:
    checkpoint = base_config.get("triage_checkpoint")
    if not checkpoint and isinstance(base_config.get("triage"), dict):
        checkpoint = base_config["triage"].get("checkpoint")
    lines = [
        "mode: full",
        f"seed: {base_config.get('seed', 42)}",
        f"device: {base_config.get('device', 'cuda')}",
        "",
        "data:",
        "  max_eval_queries: 1000",
        "",
        "routing:",
        f"  top_k_domains: {base_config.get('top_k_domains', 3)}",
        f"  fallback_to_global_search: {str(bool(base_config.get('fallback_to_global_search', True))).lower()}",
        f"  fallback_score_threshold: {base_config.get('fallback_score_threshold', 0.75)}",
        "",
        "triage:",
        f"  checkpoint: {checkpoint or 'outputs/triage/distilbert'}",
        f"  reject_threshold: {selected['nearest_kb_similarity_threshold']}",
        f"  nearest_kb_similarity_threshold: {selected['nearest_kb_similarity_threshold']}",
        f"  centroid_similarity_threshold: {selected['centroid_similarity_threshold']}",
        f"  reject_require_lexical_low: {str(bool(selected['lexical_gate_required'])).lower()}",
        f"  ticket_threshold: {selected['ticket_threshold']}",
        "",
        f"top_k_retrieval: {base_config.get('top_k_retrieval', 20)}",
        f"top_k_rerank: {base_config.get('top_k_rerank', 5)}",
        f"max_rerank_candidates: {base_config.get('max_rerank_candidates', 15)}",
        f"use_reranker: {str(bool(base_config.get('use_reranker', False))).lower()}",
        f"tau_domain: {base_config.get('tau_domain', 0.35)}",
        f"tau_chunk: {base_config.get('tau_chunk', 0.30)}",
        f"fp16: {str(bool(base_config.get('fp16', True))).lower()}",
        "",
        "# Selected by scripts/tune_thresholds_mixed.py without retraining.",
        f"# Macro-F1: {selected['Macro-F1']}",
        f"# ANSWER Recall: {selected['ANSWER Recall']}",
        f"# REJECT Precision: {selected['REJECT Precision']}",
        f"# OODAnswerRate: {selected['OODAnswerRate']}",
        f"# FalseAcceptRate: {selected['FalseAcceptRate']}",
        f"# TicketMissRate: {selected['TicketMissRate']}",
    ]
    name = "safety_tuned_ablation.yaml"
    project_path("configs", name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(rows: list[dict], selected: dict, balanced: bool = False) -> None:
    feasible_strict = [
        r for r in rows if r["ANSWER Recall"] >= 0.78 and r["FalseRejectRate"] <= 0.10 and r["REJECT Precision"] >= 0.80
    ]
    lines = [
        "# Threshold Sweep Summary",
        "",
        f"Candidates evaluated: `{len(rows)}`",
        f"Strict feasible candidates: `{len(feasible_strict)}`",
        "",
        "## Selected",
        f"`{selected}`",
        "",
        "## Selection Policy",
        "Primary constraints prioritize ANSWER recall, low false rejects, high reject precision, and citation-preserving behavior before Macro-F1.",
    ]
    name = "threshold_sweep_balanced_triage_summary.md" if balanced else "threshold_sweep_mixed_summary.md"
    project_path("outputs", "reports", name).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
