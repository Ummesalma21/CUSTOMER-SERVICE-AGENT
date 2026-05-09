from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_end_to_end import run_baseline, run_proposed
from src.evaluation.metrics import grounding_metrics, retrieval_metrics
from src.utils.io import load_config, project_path, read_jsonl, write_json, write_jsonl


LABELS = ["ANSWER", "TICKET", "REJECT"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_eval_calibrated.yaml")
    parser.add_argument("--mixed-path", default="data/processed/eval_mixed_1000.jsonl")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    prefix = args.output_prefix or _default_prefix(args.config)
    log_path = project_path("outputs", "logs", f"{prefix}_eval.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = load_config(args.config)
    mixed_path = project_path(*Path(args.mixed_path).parts)
    if args.rebuild or not mixed_path.exists():
        rows = build_mixed_eval(seed=int(config.get("seed", 42)))
        write_jsonl(mixed_path, rows)
    rows = read_jsonl(mixed_path)
    baseline = [_as_baseline_prediction(run_baseline(row["query"], config)) for row in rows]
    proposed = [run_proposed(row["query"], config) for row in rows]
    metrics = {
        "config": args.config,
        "eval_path": args.mixed_path,
        "counts": _counts(rows),
        "baseline": _decision_metrics(rows, baseline),
        "proposed": _decision_metrics(rows, proposed),
        "answer_only_retrieval": {
            "baseline": _answer_retrieval(rows, baseline),
            "proposed": _answer_retrieval(rows, proposed),
        },
        "latency": _latency(proposed),
    }
    out_predictions = []
    for row, base, prop in zip(rows, baseline, proposed):
        out_predictions.append(
            {
                **row,
                "baseline_decision": base["decision"],
                "baseline_answer": base.get("answer", ""),
                "baseline_hits": base.get("hits", [])[:5],
                "proposed_decision": prop["decision"],
                "proposed_answer": prop.get("answer", ""),
                "proposed_latency_ms": prop.get("latency_ms", 0.0),
                "proposed_hits": prop.get("hits", [])[:5],
                "proposed_tool_trace": prop.get("tool_trace", []),
            }
        )
    write_json(project_path("outputs", "reports", f"{prefix}_metrics.json"), metrics)
    write_jsonl(project_path("outputs", "reports", f"{prefix}_predictions.jsonl"), out_predictions)
    _write_summary(metrics, prefix)
    log_path.write_text(
        f"config={args.config}\neval_path={args.mixed_path}\nrows={len(rows)}\nseconds={time.perf_counter() - started:.3f}\nmetrics={metrics}\n",
        encoding="utf-8",
    )
    print(metrics)


def _default_prefix(config_path: str) -> str:
    stem = Path(config_path).stem
    if stem == "final_eval_balanced_triage":
        return "final_mixed"
    if stem == "final_eval_balanced_triage_best":
        return "final_mixed_best"
    return "mixed_eval"


def build_mixed_eval(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    eval_rows = read_jsonl(project_path("data", "processed", "eval_set.jsonl"))
    fallback_answer_rows = read_jsonl(project_path("data", "processed", "dialogue_turns.jsonl"))
    answer_rows = []
    seen_queries = set()
    for row in list(eval_rows) + list(fallback_answer_rows):
        if not row.get("gold_chunk_id") or not row.get("query") or row["query"] in seen_queries:
            continue
        if not _usable_answer_query(row["query"]):
            continue
        seen_queries.add(row["query"])
        answer_rows.append(
            {
                "query_id": f"mixed_answer_{len(answer_rows):04d}",
                "query": row["query"],
                "gold_decision": "ANSWER",
                "gold_triage": "ANSWER",
                "gold_domain": row.get("gold_domain"),
                "gold_doc_id": row.get("gold_doc_id"),
                "gold_chunk_id": row.get("gold_chunk_id"),
                "source_type": "multidoc2dial",
            }
        )
        if len(answer_rows) == 600:
            break
    if len(answer_rows) < 600:
        raise SystemExit(f"Need 600 answer rows, found {len(answer_rows)}")
    ticket_rows = _synthetic_ticket_rows(200)
    reject_rows = _synthetic_reject_rows(200)
    rows = answer_rows + ticket_rows + reject_rows
    rng.shuffle(rows)
    return rows


def _usable_answer_query(query: str) -> bool:
    q = (query or "").strip().lower()
    words = [w for w in q.replace("/", " ").split() if w.strip(".,?!;:")]
    if len(words) < 5:
        return False
    bad_starts = ("yes,", "no,", "not,", "sorry", "no sorry", "yes.", "no.")
    if q.startswith(bad_starts):
        return False
    vague = {
        "what are the different ones",
        "what is the on-site investigation about",
        "i need more information",
        "i find myself needing more information",
        "tell me more",
    }
    if any(item in q for item in vague):
        return False
    support_terms = {
        "benefit", "benefits", "ssi", "social security", "dmv", "license", "registration", "vehicle",
        "veteran", "va", "claim", "student", "loan", "fafsa", "renew", "renewal", "card", "document",
        "appeal", "application", "eligible", "payment", "address", "online",
    }
    has_support_term = any(term in q for term in support_terms)
    has_question_shape = "?" in q or q.startswith(("how ", "what ", "when ", "where ", "can ", "do ", "does ", "is ", "are ", "will ", "should ", "why "))
    return has_support_term and has_question_shape


def _synthetic_ticket_rows(n: int) -> list[dict]:
    domains = {
        "ssa": [
            "My benefits renewal is stuck as pending for case {case_id}; can someone check my account?",
            "I submitted income documents for my SSI benefits, but the portal says manual review. What should support do next?",
            "My Social Security benefit verification letter has the wrong address. Can you update my record?",
            "The benefits portal accepted my renewal but no confirmation number appeared for account {case_id}.",
        ],
        "dmv": [
            "My DMV renewal payment was charged twice for transaction {case_id}; can you review it?",
            "I uploaded my insurance proof but my registration renewal still says suspended on my account.",
            "My license renewal appointment was canceled and the portal will not let me reschedule for case {case_id}.",
            "The title status page says manual review for my vehicle. Can support check my record?",
        ],
        "va": [
            "My VA burial eligibility application says pending for claim {case_id}; can someone inspect my file?",
            "I uploaded discharge papers, but the VA portal still says documents missing on my account.",
            "My dependent information is wrong in the VA benefits portal. Can support correct it?",
            "The VA form submission failed after payment authorization for case {case_id}.",
        ],
        "studentaid": [
            "My FAFSA correction is locked on my account and I cannot submit the requested update.",
            "Student aid says my loan recertification needs manual review for case {case_id}.",
            "My school is missing from my aid application even though I added it yesterday.",
            "The income-driven repayment form errored after I uploaded tax documents.",
        ],
    }
    rows = []
    i = 0
    while len(rows) < n:
        for domain, templates in domains.items():
            for template in templates:
                if len(rows) >= n:
                    break
                rows.append(
                    {
                        "query_id": f"mixed_ticket_{i:04d}",
                        "query": template.format(case_id=f"ACCT-{100000 + i}"),
                        "gold_decision": "TICKET",
                        "gold_triage": "TICKET",
                        "gold_domain": domain,
                        "gold_doc_id": None,
                        "gold_chunk_id": None,
                        "source_type": "synthetic_ticket",
                    }
                )
                i += 1
    return rows


def _synthetic_reject_rows(n: int) -> list[dict]:
    easy = [
        "Who won the IPL yesterday?",
        "What is the weather in Mumbai tomorrow?",
        "Summarize the latest Apple stock earnings.",
        "Write me a Python script to scrape Instagram.",
        "Who is leading the Premier League table today?",
        "What movie should I watch tonight?",
    ]
    hard_support_like = [
        "Can I renew my Costco membership online?",
        "My Amazon return says pending, can you check my refund?",
        "How do I appeal a private health insurance denial?",
        "Can United Airlines change my ticket after check-in?",
        "My bank locked my debit card; can support unlock it?",
        "How do I renew a Canadian passport online?",
    ]
    near_boundary = [
        "Can I renew unemployment benefits in Canada through this portal?",
        "Does this support desk handle UK driving licence renewals?",
        "Can you verify my Medicare Advantage claim status?",
        "Where do I renew a university student aid scholarship?",
        "Can this DMV knowledge base answer questions about Texas vehicle inspections?",
        "How do I update veterans benefits in Australia?",
    ]
    templates = easy + hard_support_like + near_boundary
    rows = []
    for i in range(n):
        query = templates[i % len(templates)]
        if i >= len(templates):
            query = f"{query} Reference {i:03d}."
        rows.append(
            {
                "query_id": f"mixed_reject_{i:04d}",
                "query": query,
                "gold_decision": "REJECT",
                "gold_triage": "REJECT",
                "gold_domain": None,
                "gold_doc_id": None,
                "gold_chunk_id": None,
                "source_type": "synthetic_reject",
            }
        )
    return rows


def _as_baseline_prediction(prediction: dict) -> dict:
    out = dict(prediction)
    out["decision"] = "ANSWER"
    return out


def _counts(rows: Iterable[dict]) -> dict:
    out = {"total": 0}
    for row in rows:
        out["total"] += 1
        out[row["gold_decision"]] = out.get(row["gold_decision"], 0) + 1
        out[row["source_type"]] = out.get(row["source_type"], 0) + 1
    return out


def _decision_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    total = max(1, len(rows))
    out = {"Tool Decision Accuracy": sum(r["gold_decision"] == p["decision"] for r, p in zip(rows, predictions)) / total}
    f1s = []
    for label in LABELS:
        tp = sum(r["gold_decision"] == label and p["decision"] == label for r, p in zip(rows, predictions))
        fp = sum(r["gold_decision"] != label and p["decision"] == label for r, p in zip(rows, predictions))
        fn = sum(r["gold_decision"] == label and p["decision"] != label for r, p in zip(rows, predictions))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        out[f"{label} F1"] = f1
        out[f"{label} Precision"] = precision
        out[f"{label} Recall"] = recall
        f1s.append(f1)
    out["confusion_matrix"] = {
        gold: {pred: sum(r["gold_decision"] == gold and p["decision"] == pred for r, p in zip(rows, predictions)) for pred in LABELS}
        for gold in LABELS
    }
    out["Macro-F1"] = sum(f1s) / len(f1s)
    non_rejects = [(r, p) for r, p in zip(rows, predictions) if r["gold_decision"] != "REJECT"]
    rejects = [(r, p) for r, p in zip(rows, predictions) if r["gold_decision"] == "REJECT"]
    tickets = [(r, p) for r, p in zip(rows, predictions) if r["gold_decision"] == "TICKET"]
    out["FalseRejectRate"] = sum(p["decision"] == "REJECT" for _, p in non_rejects) / max(1, len(non_rejects))
    out["FalseAcceptRate"] = sum(p["decision"] != "REJECT" for _, p in rejects) / max(1, len(rejects))
    out["OODAnswerRate"] = sum(p["decision"] == "ANSWER" for _, p in rejects) / max(1, len(rejects))
    out["TicketMissRate"] = sum(p["decision"] != "TICKET" for _, p in tickets) / max(1, len(tickets))
    return out


def _answer_retrieval(rows: list[dict], predictions: list[dict]) -> dict:
    pairs = [(r, p) for r, p in zip(rows, predictions) if r["gold_decision"] == "ANSWER"]
    answer_rows = [r for r, _ in pairs]
    answer_predictions = [p for _, p in pairs]
    out = retrieval_metrics(answer_rows, answer_predictions, k=5)
    out.pop("Recall@1", None)
    out.update({"CitationPrecision": grounding_metrics(answer_predictions)["CitationPrecision"]})
    return out


def _latency(predictions: list[dict]) -> dict:
    times = sorted(float(p.get("latency_ms", 0.0)) for p in predictions)
    if not times:
        return {"avg_ms": 0.0, "p95_ms": 0.0, "qps": 0.0}
    avg = sum(times) / len(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    return {"avg_ms": avg, "p95_ms": p95, "qps": 1000.0 / avg if avg else 0.0}


def _write_summary(metrics: dict, prefix: str) -> None:
    lines = [
        "# Mixed Evaluation Summary",
        "",
        f"Config: `{metrics['config']}`",
        f"Eval file: `{metrics['eval_path']}`",
        "",
        "## Counts",
        f"`{metrics['counts']}`",
        "",
        "## Baseline RAG",
        f"`{metrics['baseline']}`",
        "",
        "## Proposed Calibrated System",
        f"`{metrics['proposed']}`",
        "",
        "## ANSWER-Only Retrieval",
        f"Baseline: `{metrics['answer_only_retrieval']['baseline']}`",
        f"Proposed: `{metrics['answer_only_retrieval']['proposed']}`",
        "",
        "## Latency",
        f"`{metrics.get('latency', {})}`",
        "",
        "## Notes",
        "Baseline RAG is answer-only, so TICKET and REJECT examples are counted as decision errors.",
        "This mixed evaluation does not retrain any model; it reuses the configured checkpoints and calibrated inference logic.",
    ]
    project_path("outputs", "reports", f"{prefix}_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
