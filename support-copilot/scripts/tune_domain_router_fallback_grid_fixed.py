from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_mixed import _decision_metrics
from scripts.evaluate_three_way_final import _answer_metrics, _load_esa_embedder
from src.generation.templates import cited_answer, reject_answer, ticket_answer
from src.retrieval.search_kb import search
from src.routing.domain_router import route_query
from src.utils.io import load_config, project_path, read_json, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed fast fallback grid for 4-domain router.")
    parser.add_argument("--config", default="configs/domain_router_experiment.yaml")
    parser.add_argument("--val-path", default="data/processed/domain_router_val_clean.jsonl")
    parser.add_argument("--test-path", default="data/processed/domain_router_test_eval.jsonl")
    args = parser.parse_args()
    t0 = time.perf_counter()

    config = load_config(args.config)
    val_rows = read_jsonl(project_path(*Path(args.val_path).parts))
    test_rows = read_jsonl(project_path(*Path(args.test_path).parts))
    if not val_rows or not test_rows:
        raise SystemExit("Missing validation/test rows.")

    # Build 12 combos: top_k_domains(3) x threshold bundles(4)
    bundles = [
        {"min_domain_confidence": 0.4, "min_candidate_similarity": 0.3, "min_domain_candidates": 5},
        {"min_domain_confidence": 0.5, "min_candidate_similarity": 0.3, "min_domain_candidates": 20},
        {"min_domain_confidence": 0.7, "min_candidate_similarity": 0.4, "min_domain_candidates": 5},
        {"min_domain_confidence": 0.7, "min_candidate_similarity": 0.4, "min_domain_candidates": 20},
    ]
    combos = []
    for k in [1, 2, 3]:
        for b in bundles:
            combos.append({"top_k_domains": k, **b, "enable_global_fallback": True, "fallback_merge_mode": "merge", "rerank_after_merge": True})

    cache_val = _build_cache(val_rows, config, top_k_domains=3)
    cache_test = _build_cache(test_rows, config, top_k_domains=3)
    proposed = read_json(project_path("outputs", "reports", "baseline_vs_proposed_metrics.json"), {})
    proposed_answer = (proposed.get("answer_only") or {}).get("proposed", {})
    proposed_safety = (proposed.get("unsupported_answer_safety") or {}).get("proposed", {})

    grid_rows = []
    best = None
    for i, combo in enumerate(combos, start=1):
        val_metrics = _eval_with_cache(val_rows, cache_val, combo)
        row = {
            **combo,
            "avg_num_domains_searched": val_metrics["diag"]["avg_num_domains_searched"],
            "avg_candidate_count_pre_rerank": val_metrics["diag"]["avg_candidate_count_pre_rerank"],
            "fallback_trigger_count": val_metrics["diag"]["fallback_trigger_count"],
            "avg_domain_confidence": val_metrics["diag"]["avg_domain_confidence"],
            "gold_domain_in_searched_pct": val_metrics["diag"]["gold_domain_in_searched_pct"],
            "Recall@5": val_metrics["answer_only"]["Recall@5"],
            "EvidenceHit@5": val_metrics["answer_only"]["EvidenceHit@5"],
            "UnsupportedAnswerRate": val_metrics["unsupported_answer_safety"]["UnsupportedAnswerRate"],
            "OODAnswerRate": val_metrics["unsupported_answer_safety"]["OODAnswerRate"],
            "TicketMissRate": val_metrics["unsupported_answer_safety"]["TicketMissRate"],
            "AQS": val_metrics["answer_only"]["AQS"],
            "Macro-F1": val_metrics["mixed_workflow"]["Macro-F1"],
        }
        grid_rows.append(row)
        print(
            f"[grid {i}/12] topk={row['top_k_domains']} conf={row['min_domain_confidence']} sim={row['min_candidate_similarity']} minc={row['min_domain_candidates']} "
            f"R5={row['Recall@5']:.4f} EH5={row['EvidenceHit@5']:.4f} AQS={row['AQS']:.4f} UAR={row['UnsupportedAnswerRate']:.4f} fb={row['fallback_trigger_count']}",
            flush=True,
        )

    feasible = [
        r
        for r in grid_rows
        if r["UnsupportedAnswerRate"] <= float(proposed_safety.get("UnsupportedAnswerRate", 1.0))
        and r["TicketMissRate"] <= float(proposed_safety.get("TicketMissRate", 1.0))
        and r["OODAnswerRate"] <= float(proposed_safety.get("OODAnswerRate", 1.0))
    ]
    pool = feasible if feasible else grid_rows
    pool.sort(key=lambda r: (r["Recall@5"], r["EvidenceHit@5"], r["AQS"]), reverse=True)
    best = pool[0]

    test_metrics = _eval_with_cache(test_rows, cache_test, best)
    selection = {
        "recall_at5_min": test_metrics["answer_only"]["Recall@5"] >= 0.3420,
        "evidencehit_at5_min": test_metrics["answer_only"]["EvidenceHit@5"] >= 0.3420,
        "aqs_min": test_metrics["answer_only"]["AQS"] >= 0.6733,
    }

    out = {
        "config": args.config,
        "runtime_sec": time.perf_counter() - t0,
        "grid_rows": grid_rows,
        "selected_validation_setting": best,
        "test_metrics": test_metrics,
        "proposed_reference": {
            "Recall@5": proposed_answer.get("Recall@5"),
            "EvidenceHit@5": proposed_answer.get("EvidenceHit@5"),
            "AQS": proposed_answer.get("AQS"),
            "UnsupportedAnswerRate": proposed_safety.get("UnsupportedAnswerRate"),
            "OODAnswerRate": proposed_safety.get("OODAnswerRate"),
            "TicketMissRate": proposed_safety.get("TicketMissRate"),
        },
        "selection_rule_checks": selection,
        "recommendation": "consider_final" if all(selection.values()) else "ablation_or_future_work",
    }

    write_json(project_path("outputs", "reports", "domain_router_fallback_grid_fixed_metrics.json"), out)
    write_jsonl(project_path("outputs", "reports", "domain_router_fallback_grid_fixed_predictions.jsonl"), test_metrics["predictions"])
    project_path("outputs", "reports", "domain_router_fallback_grid_fixed_summary.md").write_text(_summary(out), encoding="utf-8")
    print(json.dumps(out, indent=2))


def _build_cache(rows: list[dict], config: dict, top_k_domains: int) -> list[dict]:
    c = deepcopy(config)
    c.setdefault("domain_router", {})
    c["domain_router"]["top_k_domains"] = top_k_domains
    c["domain_router"]["enable_global_fallback"] = True
    cache = []
    for r in rows:
        q = r.get("query", "")
        routed = route_query(q, r.get("history", ""), c)
        kb_domains = [d for d in (routed.get("kb_domains") or []) if d][:top_k_domains]
        domain_hits = {d: search(q, top_k=20, domain=d) for d in kb_domains}
        global_hits = search(q, top_k=20, domain=None)
        cache.append({"routed": routed, "kb_domains": kb_domains, "domain_hits": domain_hits, "global_hits": global_hits})
    return cache


def _eval_with_cache(rows: list[dict], cache: list[dict], combo: dict) -> dict:
    preds = []
    diag_domains = []
    diag_cands = []
    diag_fb = 0
    diag_conf = []
    diag_goldin = 0
    for row, c in zip(rows, cache):
        routed = c["routed"]
        kb_domains = c["kb_domains"][: int(combo["top_k_domains"])]
        hits = []
        for d in kb_domains:
            hits.extend(c["domain_hits"].get(d, []))
        hits = _merge(hits)
        best_score = float(hits[0]["score"]) if hits else 0.0
        conf = float(routed.get("router_confidence", 0.0))
        fallback = False
        if len(hits) < int(combo["min_domain_candidates"]) or best_score < float(combo["min_candidate_similarity"]) or conf < float(combo["min_domain_confidence"]):
            fallback = True
            hits = _merge(hits + c["global_hits"])
        top_hits = hits[:5]
        action = (routed.get("action") or {}).get("label", "ANSWER")
        if action == "REJECT":
            decision, answer, cites = "REJECT", reject_answer(), []
        elif action == "TICKET":
            decision, answer, cites = "TICKET", ticket_answer({"ticket_id": "GRIDFIX"}, (routed.get("domain") or {}).get("label", "support")), []
        else:
            decision, answer, cites = "ANSWER", cited_answer(row.get("query", ""), top_hits), top_hits[:1]
        preds.append({"decision": decision, "answer": answer, "hits": top_hits, "citations": cites, "fallback": fallback, "searched_domains": kb_domains})

        diag_domains.append(len(kb_domains))
        diag_cands.append(len(hits))
        diag_fb += int(fallback)
        diag_conf.append(conf)
        gd = str(row.get("gold_domain_4way") or "").upper()
        topd = [str(x).upper() for x in (routed.get("top_domains") or [])[: int(combo["top_k_domains"])]]
        diag_goldin += int(bool(gd) and gd in topd)

    answer_rows = [r for r in rows if r.get("gold_chunk_id")]
    answer_preds = [p for r, p in zip(rows, preds) if r.get("gold_chunk_id")]
    mixed_rows = [r for r in rows if r.get("gold_decision") in {"ANSWER", "TICKET", "REJECT"}]
    mixed_preds = [p for r, p in zip(rows, preds) if r.get("gold_decision") in {"ANSWER", "TICKET", "REJECT"}]
    embedder = _load_esa_embedder()
    answer_metrics = _answer_metrics(answer_rows, answer_preds, embedder)
    mixed_metrics = _decision_metrics(mixed_rows, mixed_preds)
    safety = _safety(mixed_rows, mixed_preds)
    return {
        "answer_only": {k: answer_metrics.get(k) for k in ["Recall@5", "EvidenceHit@5", "ESA", "AQS"]},
        "mixed_workflow": {k: mixed_metrics.get(k) for k in ["Macro-F1", "ANSWER F1", "TICKET F1", "REJECT F1"]},
        "unsupported_answer_safety": safety,
        "diag": {
            "avg_num_domains_searched": sum(diag_domains) / max(1, len(diag_domains)),
            "avg_candidate_count_pre_rerank": sum(diag_cands) / max(1, len(diag_cands)),
            "fallback_trigger_count": diag_fb,
            "avg_domain_confidence": sum(diag_conf) / max(1, len(diag_conf)),
            "gold_domain_in_searched_pct": diag_goldin / max(1, len(rows)),
        },
        "predictions": [{**r, "predicted_decision": p.get("decision"), "searched_domains": p.get("searched_domains"), "fallback": p.get("fallback")} for r, p in zip(rows, preds)],
    }


def _merge(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for h in sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True):
        k = h.get("chunk_id")
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


def _safety(rows: list[dict], preds: list[dict]) -> dict:
    unsupported = [r for r in rows if r.get("gold_decision") in {"TICKET", "REJECT"}]
    uns_ans = sum(r.get("gold_decision") in {"TICKET", "REJECT"} and p.get("decision") == "ANSWER" for r, p in zip(rows, preds))
    return {
        "UnsupportedAnswerRate": uns_ans / max(1, len(unsupported)),
        "OODAnswerRate": _rate(rows, preds, "REJECT", "ANSWER"),
        "TicketMissRate": _rate(rows, preds, "TICKET", "ANSWER"),
    }


def _rate(rows: list[dict], preds: list[dict], gold: str, pred: str) -> float:
    pairs = [(r, p) for r, p in zip(rows, preds) if r.get("gold_decision") == gold]
    return sum(p.get("decision") == pred for _, p in pairs) / max(1, len(pairs))


def _summary(payload: dict) -> str:
    b = payload["selected_validation_setting"]
    t = payload["test_metrics"]
    lines = [
        "# Domain Router Fallback Grid (Fixed)",
        "",
        "## Selected Validation Setting",
        "",
        f"- top_k_domains: `{b['top_k_domains']}`",
        f"- min_domain_confidence: `{b['min_domain_confidence']}`",
        f"- min_candidate_similarity: `{b['min_candidate_similarity']}`",
        f"- min_domain_candidates: `{b['min_domain_candidates']}`",
        "",
        "## Clean Test Metrics",
        "",
        f"- Recall@5: `{t['answer_only']['Recall@5']:.4f}`",
        f"- EvidenceHit@5: `{t['answer_only']['EvidenceHit@5']:.4f}`",
        f"- AQS: `{t['answer_only']['AQS']:.4f}`",
        f"- Macro-F1: `{t['mixed_workflow']['Macro-F1']:.4f}`",
        f"- UnsupportedAnswerRate: `{t['unsupported_answer_safety']['UnsupportedAnswerRate']:.4f}`",
        f"- OODAnswerRate: `{t['unsupported_answer_safety']['OODAnswerRate']:.4f}`",
        f"- TicketMissRate: `{t['unsupported_answer_safety']['TicketMissRate']:.4f}`",
        "",
        f"Recommendation: `{payload['recommendation']}`",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
