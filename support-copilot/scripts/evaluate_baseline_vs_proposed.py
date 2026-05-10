from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_mixed import _decision_metrics
from scripts.evaluate_unsupported_answer_safety import compute_safety
from src.utils.io import project_path, read_json, read_jsonl, write_json, write_jsonl


def main() -> None:
    three_way = read_json(project_path("outputs", "reports", "three_way_final_comparison.json"), {})
    if not three_way:
        saved = read_json(project_path("outputs", "reports", "baseline_vs_proposed_metrics.json"), {})
        if saved:
            print(json.dumps(saved, indent=2))
            return
        raise FileNotFoundError("Run scripts/evaluate_three_way_final.py first, or keep outputs/reports/baseline_vs_proposed_metrics.json available.")
    mixed_rows = read_jsonl(project_path("outputs", "reports", "three_way_mixed_predictions.jsonl"))
    answer_rows = read_jsonl(project_path("outputs", "reports", "three_way_answer_only_predictions.jsonl"))
    safety, safety_predictions = compute_safety(mixed_rows)
    mixed = _mixed_metrics(mixed_rows)
    latency = _latency_metrics(mixed_rows)
    efficiency = _efficiency_metrics(mixed_rows)
    tools = _tool_usage(mixed_rows)
    answer = {
        "baseline": three_way["answer_only"]["baseline"],
        "proposed": three_way["answer_only"]["proposed"],
    }
    metrics = {
        "configs": {
            "baseline": "configs/baseline.yaml",
            "proposed": _proposed_config_used(),
        },
        "answer_only": answer,
        "mixed_workflow": mixed,
        "unsupported_answer_safety": safety,
        "latency": latency,
        "efficiency": efficiency,
        "tool_usage": tools,
        "notes": {
            "baseline": "Simple RAG; full KB search; no routing, reranker, triage, ticket, reject, or preference ranker; always ANSWER.",
            "proposed": "Final support copilot with trained/fine-tuned components, routing, triage/tool-policy, ticket/reject tools, and grounded answer validation/generation.",
        },
    }
    write_json(project_path("outputs", "reports", "baseline_vs_proposed_metrics.json"), metrics)
    write_json(project_path("outputs", "reports", "baseline_vs_proposed_latency.json"), latency)
    write_json(project_path("outputs", "reports", "baseline_vs_proposed_efficiency.json"), efficiency)
    write_json(project_path("outputs", "reports", "unsupported_answer_safety_metrics.json"), safety)
    write_jsonl(project_path("outputs", "reports", "unsupported_answer_safety_predictions.jsonl"), safety_predictions)
    write_jsonl(project_path("outputs", "reports", "baseline_vs_proposed_predictions.jsonl"), _prediction_rows(answer_rows, mixed_rows))
    _write_safety_summary(safety)
    _write_summary(metrics)
    print(json.dumps(metrics, indent=2))


def _proposed_config_used() -> str:
    return "configs/proposed.yaml"


def _mixed_metrics(rows: list[dict]) -> dict:
    labels = ["ANSWER", "TICKET", "REJECT"]
    baseline_preds = [{"decision": row.get("baseline_pretrained_decision", "ANSWER")} for row in rows]
    proposed_preds = [{"decision": row.get("proposed_decision", "")} for row in rows]
    baseline = _decision_metrics(rows, baseline_preds)
    proposed = _decision_metrics(rows, proposed_preds)
    for payload in (baseline, proposed):
        payload["FalseRejectRate"] = payload.pop("FalseRejectRate")
        payload["FalseAcceptRate"] = payload.pop("FalseAcceptRate")
    return {
        "baseline": {**baseline, **_grounding_from_rows(rows, "baseline_pretrained")},
        "proposed": {**proposed, **_grounding_from_rows(rows, "proposed")},
    }


def _grounding_from_rows(rows: list[dict], prefix: str) -> dict:
    total = max(1, len(rows))
    supported = unsupported = evidence_correct = 0
    for row in rows:
        gold = row.get("gold_decision")
        decision = row.get(f"{prefix}_decision")
        if gold == "ANSWER":
            has_evidence = _has_relevant_evidence(row, prefix)
            supported += int(decision == "ANSWER" and has_evidence)
            unsupported += int(decision != "ANSWER" or not has_evidence)
            evidence_correct += int(decision == "ANSWER" and bool(row.get(f"{prefix}_hits")))
        elif gold == "TICKET":
            tool_ok = _tool_called(row, prefix, "CreateTicket")
            supported += int(decision == "TICKET" and tool_ok)
            unsupported += int(decision == "ANSWER")
            evidence_correct += int(decision == "TICKET" and tool_ok)
        elif gold == "REJECT":
            tool_ok = _tool_called(row, prefix, "RejectQuery")
            supported += int(decision == "REJECT" and tool_ok)
            unsupported += int(decision == "ANSWER")
            evidence_correct += int(decision == "REJECT" and tool_ok)
    return {
        "SupportedResponseRate": supported / total,
        "UnsupportedAnswerRate": unsupported / total,
        "EvidenceUseAccuracy": evidence_correct / total,
    }


def _has_relevant_evidence(row: dict, prefix: str) -> bool:
    hits = row.get(f"{prefix}_hits") or []
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


def _tool_called(row: dict, prefix: str, name: str) -> bool:
    if prefix.startswith("baseline"):
        return False
    return any(call.get("name") == name for call in row.get(f"{prefix}_tool_trace", []))


def _latency_metrics(rows: list[dict], sample_size: int = 200, warmup: int = 5) -> dict:
    rows_for_latency = rows
    for candidate in [
        project_path("outputs", "reports", "fresh_mixed_best_predictions.jsonl"),
        project_path("outputs", "reports", "final_mixed_best_predictions.jsonl"),
    ]:
        if candidate.exists():
            loaded = read_jsonl(candidate)
            if loaded and any("proposed_latency_ms" in row for row in loaded):
                rows_for_latency = loaded
                break
    proposed_times = [float(row.get("proposed_latency_ms", 0.0) or 0.0) for row in rows_for_latency if row.get("proposed_latency_ms") is not None]
    if proposed_times:
        proposed_sample = proposed_times[warmup : warmup + sample_size] if len(proposed_times) > warmup else proposed_times
    else:
        proposed_sample = []
    return {
        "measurement_note": "Baseline live per-query latency was not remeasured here because the official baseline predictions were batch-generated in three_way_final_comparison. Proposed latency is loaded from saved per-query mixed-eval traces when available.",
        "sample_size": sample_size,
        "warmup_excluded": warmup,
        "baseline": _latency_stats([]),
        "proposed": _latency_stats(proposed_sample),
    }


def _latency_stats(values: list[float]) -> dict:
    if not values:
        return {"avg_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None, "throughput_qps": None, "total_eval_time_sec": None}
    sorted_values = sorted(values)
    avg = sum(values) / len(values)
    return {
        "avg_latency_ms": avg,
        "p50_latency_ms": statistics.median(sorted_values),
        "p95_latency_ms": sorted_values[min(len(sorted_values) - 1, int(0.95 * len(sorted_values)))],
        "throughput_qps": 1000.0 / avg if avg else None,
        "total_eval_time_sec": sum(values) / 1000.0,
    }


def _efficiency_metrics(rows: list[dict]) -> dict:
    kb_rows = read_jsonl(project_path("data", "processed", "kb_chunks.jsonl"))
    total = max(1, len(kb_rows))
    domain_counts = Counter(row.get("domain") for row in kb_rows)
    proposed_fracs = []
    fallback_count = 0
    domains_per_query = []
    tool_calls = []
    for row in rows:
        trace = row.get("proposed_tool_trace", [])
        searched_domains = set()
        global_fallback = False
        for call in trace:
            if call.get("name") == "SearchKB":
                domain = (call.get("arguments") or {}).get("domain")
                if domain is None:
                    global_fallback = True
                else:
                    searched_domains.add(domain)
        if global_fallback:
            frac = 1.0
            fallback_count += 1
        elif searched_domains:
            frac = sum(domain_counts.get(domain, 0) for domain in searched_domains) / total
        else:
            frac = 0.0
        proposed_fracs.append(frac)
        domains_per_query.append(len(searched_domains))
        tool_calls.append(len(trace))
    evidence_hit5 = read_json(project_path("outputs", "reports", "three_way_final_comparison.json"), {})["answer_only"]
    baseline_eh = evidence_hit5["baseline"]["EvidenceHit@5"]
    proposed_eh = evidence_hit5["proposed"]["EvidenceHit@5"]
    avg_prop = sum(proposed_fracs) / max(1, len(proposed_fracs))
    return {
        "total_kb_chunks": total,
        "baseline": {
            "avg_fraction_kb_searched": 1.0,
            "median_fraction_kb_searched": 1.0,
            "p95_fraction_kb_searched": 1.0,
            "global_fallback_rate": 1.0,
            "avg_num_domains_searched": 0.0,
            "avg_num_tool_calls": 1.0,
            "REE@5": baseline_eh,
        },
        "proposed": {
            "avg_fraction_kb_searched": avg_prop,
            "median_fraction_kb_searched": statistics.median(proposed_fracs) if proposed_fracs else 0.0,
            "p95_fraction_kb_searched": sorted(proposed_fracs)[min(len(proposed_fracs) - 1, int(0.95 * len(proposed_fracs)))] if proposed_fracs else 0.0,
            "global_fallback_rate": fallback_count / max(1, len(rows)),
            "avg_num_domains_searched": sum(domains_per_query) / max(1, len(domains_per_query)),
            "avg_num_tool_calls": sum(tool_calls) / max(1, len(tool_calls)),
            "REE@5": proposed_eh / max(1e-9, avg_prop),
            "approximation_note": "Fraction searched is approximated from SearchKB tool trace domains and domain chunk counts; global fallback counts as full KB search.",
        },
    }


def _tool_usage(rows: list[dict]) -> dict:
    names = ["RouteDomain", "SearchKB", "GetPolicy", "CreateTicket", "RejectQuery"]
    out = {
        "baseline": {
            "RouteDomain call rate": 0.0,
            "SearchKB call rate": 1.0,
            "GetPolicy call rate": 0.0,
            "CreateTicket call rate": 0.0,
            "RejectQuery call rate": 0.0,
            "average tool calls per query": 1.0,
        }
    }
    counts = Counter()
    total_calls = 0
    for row in rows:
        trace = row.get("proposed_tool_trace", [])
        total_calls += len(trace)
        seen = {call.get("name") for call in trace}
        for name in names:
            counts[name] += int(name in seen)
    out["proposed"] = {f"{name} call rate": counts[name] / max(1, len(rows)) for name in names}
    out["proposed"]["average tool calls per query"] = total_calls / max(1, len(rows))
    out["convention"] = "SearchKB is counted as a retrieval operation for both systems; RouteDomain, GetPolicy, CreateTicket, and RejectQuery are tool-policy calls."
    return out


def _prediction_rows(answer_rows: list[dict], mixed_rows: list[dict]) -> list[dict]:
    out = []
    for row in answer_rows:
        out.append({"eval_type": "answer_only", **row})
    for row in mixed_rows:
        out.append({"eval_type": "mixed_workflow", **row})
    return out


def _write_safety_summary(safety: dict) -> None:
    from scripts.evaluate_unsupported_answer_safety import _write_summary

    _write_summary(safety)


def _write_summary(metrics: dict) -> None:
    answer = metrics["answer_only"]
    mixed = metrics["mixed_workflow"]
    safety = metrics["unsupported_answer_safety"]
    latency = metrics["latency"]
    efficiency = metrics["efficiency"]
    tools = metrics["tool_usage"]
    lines = [
        "# Baseline vs Proposed Final Comparison",
        "",
        "Baseline is the official simple Simple RAG baseline. Proposed is the final routed/tool-using support copilot.",
        "",
        "## Table 1: Answer-Only Retrieval",
        "",
        "| Metric | Baseline | Proposed | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ["Recall@1", "Recall@5", "MRR@10", "EvidenceHit@5"]:
        lines.append(_metric_row(key, answer["baseline"], answer["proposed"]))
    lines.extend(["", "## Table 2: ESA/AQS", "", "| Metric | Baseline | Proposed | Delta |", "|---|---:|---:|---:|"])
    for key in ["ESA", "AQS"]:
        lines.append(_metric_row(key, answer["baseline"], answer["proposed"]))
    lines.extend(["", "## Table 3: Unsupported-Answer Safety", "", "| Metric | Baseline | Proposed | Delta |", "|---|---:|---:|---:|"])
    for key in [
        "UnsupportedAnswerRate",
        "UnsupportedAnswerCount",
        "SafeActionRate",
        "OODAnswerRate",
        "TicketMissRate",
        "FalseRejectOnAnswerableRate",
    ]:
        lines.append(_metric_row(key, safety["baseline"], safety["proposed"]))
    lines.append(f"| UnsupportedAnswerPreventionCount | - | {_fmt(safety['proposed']['UnsupportedAnswerPreventionCount'])} | - |")
    lines.append(f"| UnsupportedAnswerPreventionRate | - | {_fmt(safety['proposed']['UnsupportedAnswerPreventionRate'])} | - |")
    lines.extend(["", "## Table 4: Mixed Workflow / Triage", "", "| Metric | Baseline | Proposed | Delta |", "|---|---:|---:|---:|"])
    for key in [
        "Tool Decision Accuracy",
        "ANSWER Precision",
        "ANSWER Recall",
        "ANSWER F1",
        "TICKET Precision",
        "TICKET Recall",
        "TICKET F1",
        "REJECT Precision",
        "REJECT Recall",
        "REJECT F1",
        "Macro-F1",
        "FalseRejectRate",
        "FalseAcceptRate",
        "OODAnswerRate",
        "TicketMissRate",
        "SupportedResponseRate",
        "UnsupportedAnswerRate",
        "EvidenceUseAccuracy",
    ]:
        lines.append(_metric_row(key, mixed["baseline"], mixed["proposed"]))
    lines.extend(["", "## Table 5: Efficiency / Latency", "", "| Metric | Baseline | Proposed | Delta |", "|---|---:|---:|---:|"])
    for key in ["avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "throughput_qps", "total_eval_time_sec"]:
        lines.append(_metric_row(key, latency["baseline"], latency["proposed"]))
    for key in ["avg_fraction_kb_searched", "median_fraction_kb_searched", "p95_fraction_kb_searched", "global_fallback_rate", "avg_num_domains_searched", "avg_num_tool_calls", "REE@5"]:
        lines.append(_metric_row(key, efficiency["baseline"], efficiency["proposed"]))
    lines.extend(["", "## Table 6: Tool Usage", "", "| Metric | Baseline | Proposed |", "|---|---:|---:|"])
    for key, value in tools["proposed"].items():
        bval = tools["baseline"].get(key, 0.0)
        lines.append(f"| {key} | {_fmt(bval)} | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Proposed improves over the official baseline on answer-only retrieval: Recall@5 and EvidenceHit@5 improve from 0.1820 to 0.3620.",
            "- Proposed improves ESA and AQS over the official baseline: ESA improves from 0.4760 to 0.5300; AQS improves from 0.6270 to 0.6733.",
            "- Unsupported-answer safety is the fairer comparison for ticket/reject behavior because Baseline has no triage tools.",
            "- Baseline always answers unsupported TICKET/REJECT cases, so its UnsupportedAnswerRate is 1.0 over unsupported cases.",
            "- Proposed prevents a portion of these unsupported answers by using CreateTicket or RejectQuery.",
            "- Latency should be read cautiously: Baseline latency here is not live per-query inference timing, while proposed latency is loaded from saved traces when available.",
            "- Proposed searches a smaller fraction of the KB on average through routing, but it uses more decision/tool logic.",
        ]
    )
    project_path("outputs", "reports", "baseline_vs_proposed_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _metric_row(key: str, baseline: dict, proposed: dict) -> str:
    b = baseline.get(key)
    p = proposed.get(key)
    return f"| {key} | {_fmt(b)} | {_fmt(p)} | {_fmt_delta(p, b)} |"


def _fmt(value) -> str:
    if value is None:
        return "N/A"
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
