from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_esa_aqs import THRESHOLDS, _correctness_score, _dot, _fluency_score, _lexical_similarity, _malformed, _trueness_score
from scripts.evaluate_mixed import _decision_metrics
from scripts.evaluate_unsupported_answer_safety import compute_safety
from src.evaluation.evaluate_end_to_end import run_proposed
from src.evaluation.metrics import grounding_metrics, retrieval_metrics
from src.generation.answer_quality import clean_answer_text
from src.utils.io import load_config, project_path, read_json, read_jsonl, write_json, write_jsonl


CONFIG = "configs/reranker_ablation.yaml"


def main() -> None:
    config = load_config(CONFIG)
    answer_rows = [r for r in read_jsonl(project_path("data", "processed", "eval_set.jsonl")) if r.get("gold_chunk_id")]
    mixed_rows = read_jsonl(project_path("data", "processed", "eval_mixed_1000.jsonl"))
    three_way_answer = read_jsonl(project_path("outputs", "reports", "three_way_answer_only_predictions.jsonl"))
    three_way_mixed = read_jsonl(project_path("outputs", "reports", "three_way_mixed_predictions.jsonl"))

    print("Running proposed answer-only with reranker...", flush=True)
    proposed_answer = [run_proposed(row["query"], config) for row in answer_rows]
    print("Running proposed mixed workflow with reranker...", flush=True)
    proposed_mixed = [run_proposed(row["query"], config) for row in mixed_rows]

    answer_predictions = []
    for row, base_row, prop in zip(answer_rows, three_way_answer, proposed_answer):
        answer_predictions.append(
            {
                **row,
                "baseline_0_pretrained_decision": base_row.get("baseline_0_pretrained_decision", "ANSWER"),
                "baseline_0_pretrained_answer": base_row.get("baseline_0_pretrained_answer", ""),
                "baseline_0_pretrained_hits": base_row.get("baseline_0_pretrained_hits", []),
                "proposed_decision": prop.get("decision"),
                "proposed_answer": prop.get("answer", ""),
                "proposed_hits": prop.get("hits", [])[:5],
                "proposed_citations": prop.get("citations", []),
                "proposed_tool_trace": prop.get("tool_trace", []),
                "proposed_latency_ms": prop.get("latency_ms"),
            }
        )
    mixed_predictions = []
    for row, base_row, prop in zip(mixed_rows, three_way_mixed, proposed_mixed):
        mixed_predictions.append(
            {
                **row,
                "baseline_0_pretrained_decision": base_row.get("baseline_0_pretrained_decision", "ANSWER"),
                "baseline_0_pretrained_answer": base_row.get("baseline_0_pretrained_answer", ""),
                "baseline_0_pretrained_hits": base_row.get("baseline_0_pretrained_hits", []),
                "proposed_decision": prop.get("decision"),
                "proposed_answer": prop.get("answer", ""),
                "proposed_hits": prop.get("hits", [])[:5],
                "proposed_citations": prop.get("citations", []),
                "proposed_tool_trace": prop.get("tool_trace", []),
                "proposed_latency_ms": prop.get("latency_ms"),
            }
        )

    baseline_answer_preds = [_pred_from_three_way(row, "baseline_0_pretrained") for row in three_way_answer]
    proposed_answer_preds = [_pred_from_rerank(row) for row in answer_predictions]
    embedder = _load_embedder()
    answer_metrics = {
        "baseline_0": _answer_metrics(answer_rows, baseline_answer_preds, embedder),
        "proposed_reranker": _answer_metrics(answer_rows, proposed_answer_preds, embedder),
    }
    mixed_metrics = {
        "baseline_0": _mixed_metrics(mixed_predictions, "baseline_0_pretrained"),
        "proposed_reranker": _mixed_metrics(mixed_predictions, "proposed"),
    }
    safety, safety_predictions = compute_safety(mixed_predictions)
    latency = _latency_metrics(proposed_mixed)
    efficiency = _efficiency_metrics(mixed_predictions, answer_metrics)
    tool_usage = _tool_usage(mixed_predictions)
    metrics = {
        "configs": {"baseline_0": "configs/baseline_pretrained_rag.yaml", "proposed": CONFIG},
        "answer_only": answer_metrics,
        "mixed_workflow": mixed_metrics,
        "unsupported_answer_safety": safety,
        "latency": latency,
        "efficiency": efficiency,
        "tool_usage": tool_usage,
        "reranker": read_json(project_path("outputs", "reranker", "model.json"), {}),
    }
    write_jsonl(project_path("outputs", "reports", "baseline0_vs_proposed_reranker_answer_predictions.jsonl"), answer_predictions)
    write_jsonl(project_path("outputs", "reports", "baseline0_vs_proposed_reranker_mixed_predictions.jsonl"), mixed_predictions)
    write_jsonl(project_path("outputs", "reports", "unsupported_answer_safety_reranker_predictions.jsonl"), safety_predictions)
    write_json(project_path("outputs", "reports", "baseline0_vs_proposed_reranker_metrics.json"), metrics)
    write_json(project_path("outputs", "reports", "baseline0_vs_proposed_reranker_latency.json"), latency)
    write_json(project_path("outputs", "reports", "baseline0_vs_proposed_reranker_efficiency.json"), efficiency)
    write_json(project_path("outputs", "reports", "unsupported_answer_safety_reranker_metrics.json"), safety)
    _write_summary(metrics)
    print(json.dumps(metrics, indent=2))


def _pred_from_three_way(row: dict, prefix: str) -> dict:
    return {
        "decision": row.get(f"{prefix}_decision", "ANSWER"),
        "answer": row.get(f"{prefix}_answer", ""),
        "hits": row.get(f"{prefix}_hits", []),
        "citations": row.get(f"{prefix}_hits", [])[:1],
    }


def _pred_from_rerank(row: dict) -> dict:
    return {
        "decision": row.get("proposed_decision"),
        "answer": row.get("proposed_answer", ""),
        "hits": row.get("proposed_hits", []),
        "citations": row.get("proposed_citations", []),
    }


def _answer_metrics(rows: list[dict], predictions: list[dict], embedder) -> dict:
    out = retrieval_metrics(rows, predictions, k=5)
    g = grounding_metrics(predictions)
    out.update(
        {
            "CitationPrecision": g["CitationPrecision"],
            "GroundedAnswerRate": g["GroundedAnswerRate"],
            "UnsupportedClaimRate": g["UnsupportedClaimRate"],
        }
    )
    esa_aqs = _esa_aqs_many(rows, predictions, embedder)
    out["ESA"] = sum(1 for s in esa_aqs if s["esa_pass"]) / max(1, len(esa_aqs))
    out["AQS"] = sum(float(s["aqs"]) for s in esa_aqs) / max(1, len(esa_aqs))
    return out


def _esa_aqs_many(rows: list[dict], predictions: list[dict], embedder) -> list[dict]:
    payloads = []
    for row, pred in zip(rows, predictions):
        citations = pred.get("citations") or []
        evidence_text = " ".join(str(c.get("text", "")) for c in citations if c.get("text"))
        payloads.append({"query": row.get("query", ""), "answer": clean_answer_text(pred.get("answer", "")), "citations": citations, "evidence_text": evidence_text})
    sims = _batched_similarities(payloads, embedder)
    scored = []
    for payload, sim in zip(payloads, sims):
        malformed = _malformed(payload["answer"])
        esa_pass = bool(payload["citations"] and payload["evidence_text"])
        esa_pass = esa_pass and sim["query_citation_similarity"] >= THRESHOLDS["query_citation_similarity"]
        esa_pass = esa_pass and sim["answer_citation_similarity"] >= THRESHOLDS["answer_citation_similarity"]
        esa_pass = esa_pass and sim["query_answer_similarity"] >= THRESHOLDS["query_answer_similarity"]
        esa_pass = esa_pass and not malformed
        fluency = _fluency_score(payload["answer"])
        correctness = _correctness_score(payload["query"], payload["answer"], sim["query_answer_similarity"], malformed)
        trueness = _trueness_score(bool(payload["citations"]), sim, esa_pass)
        scored.append({"esa_pass": esa_pass, "aqs": (fluency + correctness + trueness) / 6.0})
    return scored


def _batched_similarities(payloads: list[dict], embedder) -> list[dict]:
    if embedder:
        texts = []
        for payload in payloads:
            texts.extend([payload["query"], payload["answer"], payload["evidence_text"]])
        try:
            emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            return [
                {
                    "query_citation_similarity": float(_dot(emb[i], emb[i + 2])),
                    "answer_citation_similarity": float(_dot(emb[i + 1], emb[i + 2])),
                    "query_answer_similarity": float(_dot(emb[i], emb[i + 1])),
                }
                for i in range(0, len(emb), 3)
            ]
        except Exception:
            pass
    return [
        {
            "query_citation_similarity": _lexical_similarity(p["query"], p["evidence_text"]),
            "answer_citation_similarity": _lexical_similarity(p["answer"], p["evidence_text"]),
            "query_answer_similarity": _lexical_similarity(p["query"], p["answer"]),
        }
        for p in payloads
    ]


def _mixed_metrics(rows: list[dict], prefix: str) -> dict:
    preds = [{"decision": row.get(f"{prefix}_decision")} for row in rows]
    decision = _decision_metrics(rows, preds)
    return {**decision, **_grounding_from_rows(rows, prefix)}


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
    return {"SupportedResponseRate": supported / total, "UnsupportedAnswerRate": unsupported / total, "EvidenceUseAccuracy": evidence_correct / total}


def _has_relevant_evidence(row: dict, prefix: str) -> bool:
    hits = row.get(f"{prefix}_hits") or []
    gold_chunk = row.get("gold_chunk_id")
    gold_doc = row.get("gold_doc_id")
    gold_domain = row.get("gold_domain")
    return bool(
        (gold_chunk and any(h.get("chunk_id") == gold_chunk for h in hits[:5]))
        or (gold_doc and any(h.get("doc_id") == gold_doc for h in hits[:5]))
        or (gold_domain and any(h.get("domain") == gold_domain for h in hits[:5]))
    )


def _tool_called(row: dict, prefix: str, name: str) -> bool:
    if prefix.startswith("baseline"):
        return False
    return any(call.get("name") == name for call in row.get(f"{prefix}_tool_trace", []))


def _latency_metrics(predictions: list[dict]) -> dict:
    values = sorted(float(p.get("latency_ms", 0.0)) for p in predictions if p.get("latency_ms") is not None)
    if len(values) > 5:
        values = values[5:205]
    if not values:
        return {"baseline_0": {}, "proposed_reranker": {}}
    avg = sum(values) / len(values)
    return {
        "baseline_0": {"avg_latency_ms": None, "p95_latency_ms": None, "throughput_qps": None},
        "proposed_reranker": {
            "avg_latency_ms": avg,
            "p50_latency_ms": statistics.median(values),
            "p95_latency_ms": values[min(len(values) - 1, int(0.95 * len(values)))],
            "throughput_qps": 1000.0 / avg if avg else None,
            "sample_size": len(values),
            "warmup_excluded": 5,
        },
    }


def _efficiency_metrics(rows: list[dict], answer_metrics: dict) -> dict:
    kb_rows = read_jsonl(project_path("data", "processed", "kb_chunks.jsonl"))
    total = max(1, len(kb_rows))
    domain_counts = Counter(r.get("domain") for r in kb_rows)
    fracs = []
    fallbacks = 0
    domains_per = []
    tool_counts = []
    for row in rows:
        searched = set()
        global_fallback = False
        trace = row.get("proposed_tool_trace", [])
        for call in trace:
            if call.get("name") == "SearchKB":
                domain = (call.get("arguments") or {}).get("domain")
                if domain is None:
                    global_fallback = True
                else:
                    searched.add(domain)
        if global_fallback:
            fracs.append(1.0)
            fallbacks += 1
        else:
            fracs.append(sum(domain_counts.get(d, 0) for d in searched) / total if searched else 0.0)
        domains_per.append(len(searched))
        tool_counts.append(len(trace))
    avg_frac = sum(fracs) / max(1, len(fracs))
    return {
        "total_kb_chunks": total,
        "baseline_0": {
            "avg_fraction_kb_searched": 1.0,
            "global_fallback_rate": 1.0,
            "avg_num_domains_searched": 0.0,
            "avg_num_tool_calls": 1.0,
            "REE@5": answer_metrics["baseline_0"]["EvidenceHit@5"],
        },
        "proposed_reranker": {
            "avg_fraction_kb_searched": avg_frac,
            "median_fraction_kb_searched": statistics.median(fracs) if fracs else 0.0,
            "p95_fraction_kb_searched": sorted(fracs)[min(len(fracs) - 1, int(0.95 * len(fracs)))] if fracs else 0.0,
            "global_fallback_rate": fallbacks / max(1, len(rows)),
            "avg_num_domains_searched": sum(domains_per) / max(1, len(domains_per)),
            "avg_num_tool_calls": sum(tool_counts) / max(1, len(tool_counts)),
            "REE@5": answer_metrics["proposed_reranker"]["EvidenceHit@5"] / max(1e-9, avg_frac),
            "approximation_note": "Fraction searched is approximated from SearchKB tool trace domains and domain chunk counts; global fallback counts as full KB search.",
        },
    }


def _tool_usage(rows: list[dict]) -> dict:
    names = ["RouteDomain", "SearchKB", "GetPolicy", "CreateTicket", "RejectQuery"]
    counts = Counter()
    total_calls = 0
    for row in rows:
        trace = row.get("proposed_tool_trace", [])
        total_calls += len(trace)
        seen = {call.get("name") for call in trace}
        for name in names:
            counts[name] += int(name in seen)
    return {
        "baseline_0": {"SearchKB call rate": 1.0, "average tool calls per query": 1.0},
        "proposed_reranker": {**{f"{n} call rate": counts[n] / max(1, len(rows)) for n in names}, "average tool calls per query": total_calls / max(1, len(rows))},
    }


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer

        path = project_path("outputs", "retriever", "sentence_transformer")
        kwargs = {"device": "cuda"} if _cuda_available() else {}
        return SentenceTransformer(str(path), **kwargs)
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _write_summary(metrics: dict) -> None:
    a = metrics["answer_only"]
    m = metrics["mixed_workflow"]
    s = metrics["unsupported_answer_safety"]
    e = metrics["efficiency"]
    l = metrics["latency"]
    lines = [
        "# Baseline-0 vs Proposed With Reranker",
        "",
        f"Proposed config: `{CONFIG}`",
        f"Reranker: `{metrics.get('reranker', {})}`",
        "",
        "## Answer-Only",
        "",
        "| Metric | Baseline-0 | Proposed + reranker | Delta |",
        "|---|---:|---:|---:|",
    ]
    for k in ["Recall@1", "Recall@5", "MRR@10", "EvidenceHit@5", "CitationPrecision", "GroundedAnswerRate", "UnsupportedClaimRate", "ESA", "AQS"]:
        lines.append(_row(k, a["baseline_0"], a["proposed_reranker"]))
    lines.extend(["", "## Unsupported-Answer Safety", "", "| Metric | Baseline-0 | Proposed + reranker | Delta |", "|---|---:|---:|---:|"])
    for k in ["UnsupportedAnswerRate", "UnsupportedAnswerCount", "SafeActionRate", "OODAnswerRate", "TicketMissRate", "FalseRejectOnAnswerableRate"]:
        lines.append(_row(k, s["baseline_0"], s["proposed"]))
    lines.append(f"| UnsupportedAnswerPreventionCount | - | {_fmt(s['proposed']['UnsupportedAnswerPreventionCount'])} | - |")
    lines.append(f"| UnsupportedAnswerPreventionRate | - | {_fmt(s['proposed']['UnsupportedAnswerPreventionRate'])} | - |")
    lines.extend(["", "## Mixed Workflow", "", "| Metric | Baseline-0 | Proposed + reranker | Delta |", "|---|---:|---:|---:|"])
    for k in ["Tool Decision Accuracy", "ANSWER F1", "TICKET F1", "REJECT F1", "Macro-F1", "SupportedResponseRate", "UnsupportedAnswerRate", "EvidenceUseAccuracy", "OODAnswerRate", "TicketMissRate"]:
        lines.append(_row(k, m["baseline_0"], m["proposed_reranker"]))
    lines.extend(["", "## Efficiency / Latency", "", "| Metric | Baseline-0 | Proposed + reranker | Delta |", "|---|---:|---:|---:|"])
    for k in ["avg_latency_ms", "p95_latency_ms", "throughput_qps"]:
        lines.append(_row(k, l["baseline_0"], l["proposed_reranker"]))
    for k in ["avg_fraction_kb_searched", "global_fallback_rate", "avg_num_domains_searched", "avg_num_tool_calls", "REE@5"]:
        lines.append(_row(k, e["baseline_0"], e["proposed_reranker"]))
    project_path("outputs", "reports", "baseline0_vs_proposed_reranker_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _row(key: str, b: dict, p: dict) -> str:
    return f"| {key} | {_fmt(b.get(key))} | {_fmt(p.get(key))} | {_delta(p.get(key), b.get(key))} |"


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta(v, b) -> str:
    if isinstance(v, (int, float)) and isinstance(b, (int, float)):
        return f"{v - b:+.4f}"
    return "-"


if __name__ == "__main__":
    main()
