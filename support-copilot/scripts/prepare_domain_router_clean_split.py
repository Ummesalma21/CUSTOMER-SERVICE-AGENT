from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.io import project_path, read_jsonl, write_json, write_jsonl


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _canonical_domain(value: str | None, row: dict) -> str:
    raw = (value or row.get("domain") or "").strip().lower()
    doc_id = str(row.get("gold_doc_id") or row.get("doc_id") or "").lower()
    if raw in {"ssa", "socialsecurity"} or "ssa" in doc_id:
        return "SSA"
    if raw in {"va", "veterans"} or "va" in doc_id:
        return "VA"
    if raw in {"studentaid", "studentaid", "sa"} or "student" in doc_id:
        return "SA"
    # In this repo, the fourth KB domain is stored as "dmv";
    # we map it to FSA for the 4-domain experiment label space.
    if raw in {"dmv", "fsa"} or "dmv" in doc_id:
        return "FSA"
    return ""


def _key(row: dict) -> tuple[str, str, str]:
    q = _norm(row.get("query", ""))
    decision = str(row.get("gold_decision") or "ANSWER").upper()
    gold = str(row.get("gold_chunk_id") or row.get("gold_doc_id") or "NONE")
    return (q, decision, gold)


def main() -> None:
    balanced_train = read_jsonl(project_path("data", "processed", "triage_train_balanced.jsonl"))
    balanced_val = read_jsonl(project_path("data", "processed", "triage_val_balanced.jsonl"))
    eval_answer = read_jsonl(project_path("data", "processed", "eval_set.jsonl"))
    eval_mixed = read_jsonl(project_path("data", "processed", "eval_mixed_1000.jsonl"))

    heldout = [row for row in (eval_answer + eval_mixed) if row.get("query")]
    heldout_q = {_norm(row.get("query", "")) for row in heldout}
    heldout_k = {_key(row) for row in heldout}

    train_candidates = [row for row in (balanced_train + balanced_val) if row.get("query")]
    dedup = {}
    for row in train_candidates:
        row2 = dict(row)
        row2["gold_domain_4way"] = _canonical_domain(row.get("gold_domain"), row)
        k = _key(row2)
        decision = str(row2.get("gold_decision") or "ANSWER").upper()
        if not row2["gold_domain_4way"] and decision != "REJECT":
            continue
        if _norm(row2["query"]) in heldout_q or k in heldout_k:
            continue
        dedup[k] = row2
    clean_all = list(dedup.values())

    val_size = max(200, int(0.1 * len(clean_all)))
    clean_val = clean_all[:val_size]
    clean_train = clean_all[val_size:]

    clean_test = []
    for row in heldout:
        row2 = dict(row)
        row2["gold_domain_4way"] = _canonical_domain(row.get("gold_domain"), row)
        clean_test.append(row2)

    train_q = {_norm(r.get("query", "")) for r in clean_train}
    val_q = {_norm(r.get("query", "")) for r in clean_val}
    test_q = {_norm(r.get("query", "")) for r in clean_test}
    train_k = {_key(r) for r in clean_train}
    val_k = {_key(r) for r in clean_val}
    test_k = {_key(r) for r in clean_test}
    overlap = {
        "train_test_query_overlap": len(train_q & test_q),
        "val_test_query_overlap": len(val_q & test_q),
        "train_test_key_overlap": len(train_k & test_k),
        "val_test_key_overlap": len(val_k & test_k),
    }

    out_train = project_path("data", "processed", "domain_router_train_clean.jsonl")
    out_val = project_path("data", "processed", "domain_router_val_clean.jsonl")
    out_test = project_path("data", "processed", "domain_router_test_eval.jsonl")
    write_jsonl(out_train, clean_train)
    write_jsonl(out_val, clean_val)
    write_jsonl(out_test, clean_test)
    report = {
        "clean_train_rows": len(clean_train),
        "clean_val_rows": len(clean_val),
        "clean_test_rows": len(clean_test),
        "overlap": overlap,
    }
    write_json(project_path("outputs", "domain_router", "clean_split_report.json"), report)
    print(json.dumps(report, indent=2))
    if any(v > 0 for v in overlap.values()):
        raise SystemExit("Leakage overlap is non-zero; refusing to proceed.")


if __name__ == "__main__":
    main()
