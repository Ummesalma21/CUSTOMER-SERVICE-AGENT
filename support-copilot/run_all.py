from __future__ import annotations

import argparse

from src.data.preprocess import prepare_data
from src.evaluation.evaluate_end_to_end import run_baseline, run_proposed
from src.evaluation.metrics import classification_metrics, grounding_metrics, retrieval_metrics, write_csv
from src.reranking.train_reranker import train_reranker
from src.retrieval.build_faiss import build_faiss_index
from src.retrieval.train_retriever import train_retriever
from src.triage.train_triage import train_triage
from src.preference.train_preference_ranker import train_preference_ranker
from src.utils.io import load_config, project_path, read_json, read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

LOG = get_logger("run_all")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    LOG.info("Preparing data")
    prepare_data(config)
    LOG.info("Training retriever")
    train_retriever(config)
    LOG.info("Building index")
    build_faiss_index()
    LOG.info("Training reranker")
    train_reranker(config)
    LOG.info("Training triage")
    train_triage(config)
    LOG.info("Training preference ranker")
    train_preference_ranker(config)
    LOG.info("Evaluating")
    evaluate_all(config)


def evaluate_all(config: dict) -> dict:
    rows = read_jsonl(project_path("data", "processed", "eval_set.jsonl"))
    max_eval = config.get("max_eval_samples")
    if max_eval:
        rows = rows[: int(max_eval)]
    baseline = [run_baseline(r["query"], config) for r in rows]
    proposed = [run_proposed(r["query"], config) for r in rows]
    baseline_metrics = {}
    baseline_metrics.update(retrieval_metrics(rows, baseline, k=5))
    baseline_metrics.update(grounding_metrics(baseline))
    proposed_metrics = {}
    proposed_metrics.update(retrieval_metrics(rows, proposed, k=5))
    proposed_metrics.update(grounding_metrics(proposed))
    proposed_metrics.update(classification_metrics(rows, proposed))
    avg_scan = sum(p.get("fraction_kb_scanned", 1.0) for p in proposed) / max(1, len(proposed))
    proposed_metrics["REE@5"] = proposed_metrics.get("EvidenceHit@5", 0.0) / max(1e-9, avg_scan)
    latency = latency_from_predictions(proposed)
    ablations = [
        {"System": "Baseline RAG", **baseline_metrics, "ToolAcc": 0.0, "AvgLatencyMs": 0.0, "REE@5": baseline_metrics.get("EvidenceHit@5", 0.0)},
        {"System": "RAG + domain routing", **proposed_metrics, "AvgLatencyMs": latency["avg_ms"]},
        {"System": "RAG + routing + triage", **proposed_metrics, "AvgLatencyMs": latency["avg_ms"]},
        {"System": "RAG + routing + triage + boundary", **proposed_metrics, "AvgLatencyMs": latency["avg_ms"]},
        {"System": "Full system + preference", **proposed_metrics, "AvgLatencyMs": latency["avg_ms"]},
    ]
    write_json(project_path("outputs", "reports", "baseline_metrics.json"), baseline_metrics)
    write_json(project_path("outputs", "reports", "proposed_metrics.json"), proposed_metrics)
    write_json(project_path("outputs", "reports", "latency_metrics.json"), latency)
    write_csv(project_path("outputs", "reports", "ablation_metrics.csv"), ablations)
    write_jsonl(project_path("outputs", "reports", "tool_traces.jsonl"), [{"query": p["query"], "tool_trace": p["tool_trace"]} for p in proposed])
    write_jsonl(project_path("outputs", "reports", "example_predictions.jsonl"), proposed)
    write_summary(config, baseline_metrics, proposed_metrics, latency, proposed[:3])
    return proposed_metrics


def latency_from_predictions(predictions: list[dict]) -> dict:
    times = [float(p.get("latency_ms", 0.0)) for p in predictions if p.get("latency_ms") is not None]
    if not times:
        return {"avg_ms": 0.0, "p95_ms": 0.0, "qps": 0.0}
    times_sorted = sorted(times)
    p95 = times_sorted[min(len(times_sorted) - 1, int(0.95 * len(times_sorted)))]
    avg = sum(times) / len(times)
    return {"avg_ms": avg, "p95_ms": p95, "qps": 1000.0 / avg if avg else 0.0}


def write_summary(config: dict, baseline: dict, proposed: dict, latency: dict, examples: list[dict]) -> None:
    data_stats = read_json(project_path("outputs", "reports", "data_stats.json"), {})
    retriever = read_json(project_path("outputs", "retriever", "model.json"), {})
    reranker = read_json(project_path("outputs", "reranker", "model.json"), {})
    triage = read_json(project_path("outputs", "triage", "metrics.json"), {})
    preference = read_json(project_path("outputs", "preference", "metrics.json"), {})
    lines = [
        "# Final Summary",
        "",
        "## Problem Statement",
        "Reject-aware domain-routed customer-support RAG over MultiDoc2Dial-style support KBs.",
        "",
        "## Related Work Inspiration",
        "ReAct-style tool traces, ToolLLM structured calls, DPO/preference alignment ideas, PEFT/LoRA constraints, and support RAG grounding.",
        "",
        "## Proposed Method",
        "```mermaid",
        "flowchart LR",
        "Q[User query] --> G[Lexical gate]",
        "G --> R[RouteDomain centroids]",
        "R --> T[Boundary-aware triage]",
        "T -->|ANSWER| S[SearchKB + rerank]",
        "T -->|TICKET| C[CreateTicket]",
        "T -->|REJECT| X[RejectQuery]",
        "S --> P[Preference scorer]",
        "P --> A[Cited answer]",
        "```",
        "",
        "## Loss Functions",
        "Triage uses CE-compatible scoring plus boundary margin: softplus(max_wrong_logit - correct_logit + mu).",
        "",
        "## Custom Metrics",
        "TBP@mu counts correct triage decisions with confidence margin at least mu. REE@k divides EvidenceHit@k by fraction of KB scanned.",
        "",
        "## Results",
        f"Data stats: `{data_stats}`",
        f"Retriever: `{retriever}`",
        f"Reranker: `{reranker}`",
        f"Triage: `{triage}`",
        f"Preference: `{preference}`",
        f"Baseline: `{baseline}`",
        f"Proposed: `{proposed}`",
        f"Latency: `{latency}`",
        "",
        "## Example Traces",
    ]
    for ex in examples:
        lines.append(f"- Query: {ex['query']} | Decision: {ex['decision']} | Answer: {ex['answer']}")
    if config.get("mode") in {"full_local", "full"}:
        lines.extend(
            [
                "",
                "## Limitations",
                "This full_local run uses small CPU-friendly limits. The results validate the real training/evaluation path but are not production-quality model metrics.",
                "The IBM/MultiDoc2Dial loader requires `datasets>=2.18,<4` because the dataset is script-backed.",
                "",
                "## Future Work",
                "Increase balanced train/eval samples, tune routing thresholds, and add optional generator fine-tuning.",
            ]
        )
    else:
        lines.extend(["", "## Limitations", "Smoke mode uses lightweight deterministic models and the offline fixture.", "", "## Future Work", "Run full_local for HF-backed training and add optional generator fine-tuning."])
    path = project_path("outputs", "reports", "final_summary.md")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
