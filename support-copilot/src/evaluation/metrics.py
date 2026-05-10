from __future__ import annotations

import csv
from pathlib import Path

from src.triage.losses import margin
from src.utils.io import ensure_dir


def retrieval_metrics(eval_rows: list[dict], predictions: list[dict], k: int = 5) -> dict:
    total = max(1, len(eval_rows))
    hit1 = hitk = mrr = 0.0
    for row, pred in zip(eval_rows, predictions):
        gold = row.get("gold_chunk_id")
        hits = pred.get("hits", [])
        ids = [h.get("chunk_id") for h in hits]
        if gold and ids[:1] == [gold]:
            hit1 += 1
        if gold and gold in ids[:k]:
            hitk += 1
            mrr += 1.0 / (ids.index(gold) + 1)
    return {"Recall@1": hit1 / total, f"Recall@{k}": hitk / total, "MRR@10": mrr / total, f"EvidenceHit@{k}": hitk / total}


def classification_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    labels = ["ANSWER", "TICKET", "REJECT"]
    total = max(1, len(rows))
    correct = sum(1 for r, p in zip(rows, predictions) if r.get("gold_triage") == p.get("decision"))
    out = {"Tool Decision Accuracy": correct / total}
    f1s = []
    for label in labels:
        tp = sum(1 for r, p in zip(rows, predictions) if r.get("gold_triage") == label and p.get("decision") == label)
        fp = sum(1 for r, p in zip(rows, predictions) if r.get("gold_triage") != label and p.get("decision") == label)
        fn = sum(1 for r, p in zip(rows, predictions) if r.get("gold_triage") == label and p.get("decision") != label)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out[f"{label} F1"] = f1
        f1s.append(f1)
    out["Macro-F1"] = sum(f1s) / len(f1s)
    rejects = [x for x in zip(rows, predictions) if x[0].get("gold_triage") == "REJECT"]
    non_rejects = [x for x in zip(rows, predictions) if x[0].get("gold_triage") != "REJECT"]
    out["False Reject Rate"] = sum(1 for _, p in non_rejects if p.get("decision") == "REJECT") / max(1, len(non_rejects))
    out["False Accept Rate"] = sum(1 for _, p in rejects if p.get("decision") != "REJECT") / max(1, len(rejects))
    for mu in [0.10, 0.15, 0.20]:
        out[f"TBP@{mu:.2f}"] = sum(
            1 for r, p in zip(rows, predictions) if r.get("gold_triage") == p.get("decision") and p.get("triage_margin", 0.0) >= mu
        ) / total
    return out


def write_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    keys = sorted({k for row in rows for k in row})
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
