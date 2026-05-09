#!/usr/bin/env python
"""
Build a balanced triage/tool-policy dataset for the Reject-Aware Domain-Routed RAG project.

Creates:
  data/processed/triage_train_balanced.jsonl
  data/processed/triage_val_balanced.jsonl
  data/processed/triage_test_balanced.jsonl
  outputs/reports/triage_balanced_dataset_summary.json

Target labels:
  ANSWER = answerable from KB evidence
  TICKET = in-domain support issue requiring manual/account-specific action
  REJECT = out-of-domain query

Usage from repo root:
  .\\.venv\\Scripts\\python.exe scripts\\build_balanced_triage_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

RNG_SEED = 42

DEFAULT_TARGETS = {
    "train": {"ANSWER": 8000, "TICKET": 5000, "REJECT": 5000},
    "val": {"ANSWER": 1000, "TICKET": 500, "REJECT": 500},
    "test": {"ANSWER": 1000, "TICKET": 500, "REJECT": 500},
}

DOMAINS = {
    "ssa": {
        "name": "Social Security / benefits",
        "terms": ["social security", "ssa", "benefits", "benefit renewal", "retirement benefits", "disability benefits", "medicare", "ssi", "benefits portal", "claim status"],
        "objects": ["benefit renewal", "disability claim", "SSI application", "Medicare enrollment", "retirement benefits application", "benefits portal account", "benefit payment"],
        "actions": ["renew my benefits", "check my benefit application", "update my benefits account", "submit my benefits form", "upload documents for my claim", "fix my benefits portal login"],
    },
    "dmv": {
        "name": "DMV / motor vehicles",
        "terms": ["dmv", "driver license", "registration", "vehicle registration", "license plate", "inspection", "emissions inspection", "custom plates", "title certificate"],
        "objects": ["vehicle registration", "license renewal", "driver license", "custom plate order", "inspection appointment", "title certificate", "registration payment"],
        "actions": ["renew my registration", "replace my license", "check my plate order", "book an inspection", "update my vehicle title", "fix my DMV payment"],
    },
    "va": {
        "name": "Veterans Affairs",
        "terms": ["va", "veterans affairs", "veteran benefits", "burial benefits", "va cemetery", "pre-need eligibility", "military service", "veteran application"],
        "objects": ["VA benefit application", "burial eligibility form", "veteran cemetery application", "military service record", "VA claim", "pre-need eligibility request"],
        "actions": ["submit my VA form", "check my VA application", "update my veteran claim", "upload service documents", "request burial eligibility", "fix my VA account issue"],
    },
    "studentaid": {
        "name": "Student aid / education finance",
        "terms": ["student aid", "fafsa", "student loan", "loan repayment", "financial aid", "pell grant", "aid application", "loan servicer"],
        "objects": ["FAFSA application", "student loan repayment plan", "Pell Grant eligibility", "financial aid account", "student loan document", "aid correction form"],
        "actions": ["check my FAFSA status", "fix my student aid account", "update my loan repayment plan", "submit financial aid documents", "correct my aid application", "appeal my aid decision"],
    },
}

EASY_OOD_TOPICS = [
    "Who won yesterday's cricket league match?", "Write a Python program for merge sort.", "Give me a pasta recipe for dinner.",
    "What is the capital of France?", "Explain quantum mechanics in simple terms.", "Create a workout plan for building abs.",
    "Who is the best football player right now?", "Summarize the latest movie releases.", "Translate this paragraph into Spanish.",
    "Write a birthday poem for my friend.", "What is the weather in Mumbai tomorrow?", "Give me investment advice for crypto.",
    "How do I install a video game mod?", "Tell me a joke about cats.", "Help me choose a gaming laptop.",
]

HARD_SUPPORT_OOD_OBJECTS = [
    "Netflix password", "Amazon package", "Flipkart refund", "bank KYC", "airline ticket", "hotel booking", "phone warranty",
    "internet router", "electricity bill", "gas connection", "food delivery order", "ride booking", "credit card charge",
    "insurance claim", "hospital appointment", "train ticket", "passport application", "PAN card address", "Aadhaar update",
    "university hostel application", "company payroll issue",
]

NEAR_BOUNDARY_OOD = [
    "Can I renew my passport online?", "How do I update my PAN card address?", "Can I apply for university financial aid through your portal?",
    "How do I book a hospital appointment?", "Can I claim private health insurance here?", "Where do I upload documents for my bank loan?",
    "Can you check my airline refund status?", "How do I replace a lost employee ID card?", "Can I update my voter ID address?",
    "How do I apply for a municipal property tax correction?", "Can you cancel my railway ticket?", "How do I renew my professional certification license?",
    "Can I get a duplicate school leaving certificate?", "Where do I submit documents for a housing society transfer?",
]

ACCOUNT_SPECIFIC_PATTERNS = [
    "Can you check the status of my {obj}?", "My {obj} was rejected. Can you fix it?",
    "I submitted my {obj} but have not received confirmation.", "My payment for {obj} failed but money was deducted.",
    "I cannot log in to my account to manage my {obj}.", "My document upload for {obj} keeps failing.",
    "I received the wrong result for my {obj}. Can someone review it?", "My appointment for {obj} was cancelled. Can you reschedule it?",
    "The portal shows an error when I try to {action}.", "I need a human to review my {obj} because the information shown is wrong.",
    "I already submitted the required documents for {obj}; why is it still pending?", "My account is locked and I cannot {action}.",
    "Can you manually update my {obj}?", "The system charged me twice for {obj}.", "I have an urgent issue with my {obj} that is not covered in the article.",
]

ANSWER_QUERY_KEYS = ["query", "question", "user_query", "utterance", "user_utterance", "text", "input", "turn_text"]
DECISION_KEYS = ["gold_decision", "decision", "label", "gold_label", "triage_label"]
DOC_KEYS = ["gold_doc_id", "doc_id", "document_id", "evidence_doc_id", "positive_doc_id"]
CHUNK_KEYS = ["gold_chunk_id", "chunk_id", "span_id", "evidence_chunk_id", "positive_chunk_id"]
DOMAIN_KEYS = ["gold_domain", "domain", "kb_domain"]

SUPPORT_LEXICON = sorted(set(term for d in DOMAINS.values() for term in (d["terms"] + d["objects"] + d["actions"])))


def stable_id(prefix: str, text: str) -> str:
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"


def norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_first(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def infer_domain_from_text(text: str) -> Optional[str]:
    low = text.lower()
    scores = {domain: sum(1 for term in info["terms"] if term.lower() in low) for domain, info in DOMAINS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def lexical_gate_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for term in SUPPORT_LEXICON if term.lower() in low)
    return min(1.0, hits / 4.0)


def row_query(row: Dict[str, Any]) -> str:
    return norm_text(get_first(row, ANSWER_QUERY_KEYS))


def row_decision(row: Dict[str, Any]) -> Optional[str]:
    value = get_first(row, DECISION_KEYS)
    if value is None:
        return None
    s = str(value).upper()
    if "ANSWER" in s:
        return "ANSWER"
    if "TICKET" in s or "ESCALATE" in s or "CREATE" in s:
        return "TICKET"
    if "REJECT" in s or "OOD" in s or "OUT_OF_DOMAIN" in s:
        return "REJECT"
    return None


def collect_answer_examples(processed_dir: Path, exclude_exact: set[str]) -> tuple[List[Dict[str, Any]], int]:
    examples, seen = [], set()
    excluded_overlap = 0
    all_jsonl = list(processed_dir.rglob("*.jsonl"))
    preferred = sorted(all_jsonl, key=lambda p: (0 if any(x in p.name.lower() for x in ["answer", "retriever", "triage", "eval", "mixed"]) else 1, str(p)))

    for path in preferred:
        if path.name in {"eval_mixed_1000.jsonl", "triage_train_balanced.jsonl", "triage_val_balanced.jsonl", "triage_test_balanced.jsonl"}:
            continue
        for row in read_jsonl(path):
            q = row_query(row)
            if len(q) < 12:
                continue
            if q.lower() in exclude_exact:
                excluded_overlap += 1
                continue
            decision = row_decision(row)
            doc_id = get_first(row, DOC_KEYS)
            chunk_id = get_first(row, CHUNK_KEYS)
            domain = get_first(row, DOMAIN_KEYS) or infer_domain_from_text(q)
            has_evidence = bool(doc_id or chunk_id or row.get("positive") or row.get("positive_doc_id"))
            if decision not in (None, "ANSWER"):
                continue
            if not has_evidence and decision != "ANSWER":
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            examples.append({
                "id": stable_id("answer", q), "query": q, "gold_decision": "ANSWER",
                "gold_domain": str(domain or "unknown").lower(), "gold_doc_id": doc_id, "gold_chunk_id": chunk_id,
                "source_type": "multidoc2dial_answer", "lexical_gate_score": lexical_gate_score(q),
                "nearest_kb_similarity": row.get("nearest_kb_similarity"), "max_centroid_similarity": row.get("max_centroid_similarity"),
                "centroid_margin": row.get("centroid_margin"), "top1_retrieval_score": row.get("top1_retrieval_score"),
                "top5_mean_retrieval_score": row.get("top5_mean_retrieval_score"),
            })
    return examples, excluded_overlap


def load_eval_queries_to_exclude(processed_dir: Path) -> set:
    exclude = set()
    for name in ["eval_mixed_1000.jsonl"]:
        for row in read_jsonl(processed_dir / name):
            q = row_query(row)
            if q:
                exclude.add(q.lower())
    return exclude


def make_ticket_examples(n: int, rng: random.Random, split: str, exclude_exact: set[str] | None = None) -> List[Dict[str, Any]]:
    rows, seen = [], set()
    exclude_exact = exclude_exact or set()
    domains = list(DOMAINS.keys())
    attempts = 0
    while len(rows) < n:
        attempts += 1
        domain = rng.choice(domains)
        info = DOMAINS[domain]
        q = rng.choice(ACCOUNT_SPECIFIC_PATTERNS).format(obj=rng.choice(info["objects"]), action=rng.choice(info["actions"]))
        q = norm_text(rng.choice(["", "", "I need help. ", "This is urgent. "]) + q + rng.choice(["", "", " Please create a support request.", " I need manual review.", " Can a human check this?"]))
        if attempts > 2500:
            q = f"{q} Reference {split}-ticket-{attempts:05d}."
        if q.lower() in seen or q.lower() in exclude_exact:
            continue
        seen.add(q.lower())
        rows.append({
            "id": stable_id(f"ticket_{split}", q), "query": q, "gold_decision": "TICKET", "gold_domain": domain,
            "gold_doc_id": None, "gold_chunk_id": None, "source_type": "synthetic_ticket_account_specific",
            "lexical_gate_score": lexical_gate_score(q), "nearest_kb_similarity": None, "max_centroid_similarity": None,
            "centroid_margin": None, "top1_retrieval_score": None, "top5_mean_retrieval_score": None,
        })
    return rows


def make_reject_examples(n: int, rng: random.Random, split: str, exclude_exact: set[str] | None = None) -> List[Dict[str, Any]]:
    rows, seen = [], set()
    exclude_exact = exclude_exact or set()
    n_easy = int(round(n * 0.20))
    n_hard = int(round(n * 0.50))
    n_near = n - n_easy - n_hard

    def add(q: str, subtype: str) -> bool:
        q = norm_text(q)
        if not q or q.lower() in seen or q.lower() in exclude_exact:
            return False
        seen.add(q.lower())
        rows.append({
            "id": stable_id(f"reject_{split}", q), "query": q, "gold_decision": "REJECT", "gold_domain": "out_of_domain",
            "gold_doc_id": None, "gold_chunk_id": None, "source_type": f"synthetic_reject_{subtype}",
            "lexical_gate_score": lexical_gate_score(q), "nearest_kb_similarity": None, "max_centroid_similarity": None,
            "centroid_margin": None, "top1_retrieval_score": None, "top5_mean_retrieval_score": None,
        })
        return True

    attempts = 0
    while sum(1 for r in rows if r["source_type"].endswith("easy")) < n_easy:
        attempts += 1
        q = rng.choice(["", "Please help: ", "Quick question: "]) + rng.choice(EASY_OOD_TOPICS)
        if attempts > 50:
            q = f"{q} Reference {split}-easy-{attempts:05d}."
        add(q, "easy")

    hard_patterns = [
        "How do I reset my {obj}?", "Can you check the status of my {obj}?", "Where should I upload documents for my {obj}?",
        "My {obj} was rejected. Can your support team fix it?", "I paid for my {obj}, but the payment failed. Can you help?",
        "Can I cancel my {obj} through this portal?", "I need a refund for my {obj}.", "Can a human agent review my {obj}?",
        "The website shows an error for my {obj}.", "What documents are required for my {obj}?",
    ]
    attempts = 0
    while sum(1 for r in rows if r["source_type"].endswith("hard_support_like")) < n_hard:
        attempts += 1
        q = rng.choice(hard_patterns).format(obj=rng.choice(HARD_SUPPORT_OOD_OBJECTS))
        if attempts > 250:
            q = f"{q} Reference {split}-hard-{attempts:05d}."
        add(q, "hard_support_like")

    near_patterns = ["{q}", "{q} I need the official policy.", "{q} Can you check my application status?", "{q} What documents do I need?", "{q} Can I do this online?"]
    attempts = 0
    while sum(1 for r in rows if r["source_type"].endswith("near_boundary")) < n_near:
        attempts += 1
        q = rng.choice(near_patterns).format(q=rng.choice(NEAR_BOUNDARY_OOD))
        if attempts > 100:
            q = f"{q} Reference {split}-near-{attempts:05d}."
        add(q, "near_boundary")

    rng.shuffle(rows)
    return rows[:n]


def sample_answer_examples(pool: List[Dict[str, Any]], n: int, rng: random.Random, split: str) -> List[Dict[str, Any]]:
    if not pool:
        raise RuntimeError("No ANSWER examples found. Run MultiDoc2Dial preprocessing first, then rerun this script.")
    picked = rng.sample(pool, n) if len(pool) >= n else [rng.choice(pool) for _ in range(n)]
    rows = []
    for i, row in enumerate(picked):
        new = dict(row)
        new["id"] = stable_id(f"answer_{split}", f"{row['query']}::{i}")
        new["split"] = split
        if len(pool) < n:
            new["oversampled_answer"] = True
        rows.append(new)
    return rows


def add_missing_fields(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    row["split"] = split
    row["label"] = row["gold_decision"]
    row.setdefault("lexical_gate_score", lexical_gate_score(row.get("query", "")))
    for key in ["nearest_kb_similarity", "max_centroid_similarity", "centroid_margin", "top1_retrieval_score", "top5_mean_retrieval_score"]:
        row.setdefault(key, None)
    return row


def count_overlap(path: str, exclude_exact: set[str]) -> int:
    return sum(1 for row in read_jsonl(Path(path)) if row_query(row).lower() in exclude_exact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--report_dir", default="outputs/reports")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    exclude_exact = load_eval_queries_to_exclude(processed_dir)
    answer_pool, excluded_answer_overlap = collect_answer_examples(processed_dir, exclude_exact)
    rng.shuffle(answer_pool)

    outputs = {}
    for split, targets in DEFAULT_TARGETS.items():
        rows = []
        rows.extend(sample_answer_examples(answer_pool, targets["ANSWER"], rng, split))
        rows.extend(make_ticket_examples(targets["TICKET"], rng, split, exclude_exact if split == "train" else set()))
        rows.extend(make_reject_examples(targets["REJECT"], rng, split, exclude_exact if split == "train" else set()))
        rows = [add_missing_fields(r, split) for r in rows]
        rng.shuffle(rows)
        path = out_dir / f"triage_{split}_balanced.jsonl"
        write_jsonl(path, rows)
        outputs[split] = {
            "path": str(path), "count": len(rows),
            "labels": dict(Counter(r["gold_decision"] for r in rows)),
            "source_types": dict(Counter(r["source_type"] for r in rows)),
        }

    summary = {
        "seed": args.seed,
        "eval_mixed_exclusion_file": str(processed_dir / "eval_mixed_1000.jsonl"),
        "eval_mixed_exclusion_query_count": len(exclude_exact),
        "excluded_overlapping_answer_source_rows": excluded_answer_overlap,
        "answer_pool_found_after_eval_exclusion": len(answer_pool),
        "train_exact_overlap_with_eval_mixed": count_overlap(outputs["train"]["path"], exclude_exact),
        "targets": DEFAULT_TARGETS,
        "outputs": outputs,
        "notes": [
            "ANSWER rows are extracted from local MultiDoc2Dial-derived processed JSONL files.",
            "TICKET rows are synthetic in-domain account-specific/manual-review cases.",
            "REJECT rows are synthetic OOD: 20% easy, 50% hard support-like, 30% near-boundary.",
            "Exact queries from data/processed/eval_mixed_1000.jsonl are excluded from ANSWER extraction when that file exists.",
            "Similarity feature columns are included as placeholders unless already present in source rows."
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "triage_balanced_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
