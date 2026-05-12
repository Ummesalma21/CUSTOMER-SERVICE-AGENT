from __future__ import annotations

import argparse
import itertools
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.utils.io import load_config, project_path, read_json, read_jsonl, write_json

DOMAINS = ["SSA", "VA", "SA", "FSA"]


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train evidence-supervised 4-domain router.")
    parser.add_argument("--config", default="configs/domain_router_experiment.yaml")
    parser.add_argument("--train-path", default="data/processed/domain_router_train_clean.jsonl")
    parser.add_argument("--val-path", default="data/processed/domain_router_val_clean.jsonl")
    parser.add_argument("--test-path", default="data/processed/domain_router_test_eval.jsonl")
    args = parser.parse_args()
    config = load_config(args.config)
    train_rows = [_label_row(r) for r in read_jsonl(project_path(*Path(args.train_path).parts)) if r.get("query")]
    val_rows = [_label_row(r) for r in read_jsonl(project_path(*Path(args.val_path).parts)) if r.get("query")]
    test_rows = [_label_row(r) for r in read_jsonl(project_path(*Path(args.test_path).parts)) if r.get("query")]
    if not train_rows or not val_rows or not test_rows:
        raise SystemExit("Missing clean domain-router train/val/test. Run scripts/prepare_domain_router_clean_split.py.")
    leakage = leakage_report(train_rows, val_rows, test_rows)
    if any(v > 0 for v in leakage.values()):
        raise SystemExit(f"Leakage detected, aborting: {json.dumps(leakage)}")

    x_train = _load_or_encode_embeddings([_router_text(r["query"], r.get("history")) for r in train_rows], "train")
    x_val = _load_or_encode_embeddings([_router_text(r["query"], r.get("history")) for r in val_rows], "val")
    print(f"[domain-router] embeddings ready in {time.perf_counter() - started:.2f}s", flush=True)

    domain_model = _fit_model(x_train, train_rows, "domain_label", DOMAINS)
    action_labels = ["ANSWER", "TICKET", "REJECT"]
    action_model = _fit_model(x_train, train_rows, "action_label", action_labels)

    payload = {
        "domain_model": domain_model,
        "domain_labels": DOMAINS,
        "action_model": action_model,
        "action_labels": action_labels,
        "model_name": "LogisticRegression over frozen retriever query embeddings (4-domain)",
    }
    out_dir = project_path("outputs", "domain_router")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(payload, f)

    domain_metrics = _classification_metrics(domain_model, x_val, val_rows, "domain_label", DOMAINS)
    action_metrics = _classification_metrics(action_model, x_val, val_rows, "action_label", action_labels)
    selected, grid_rows = _tune_thresholds(config, payload, x_val, val_rows)
    metrics = {
        "config": args.config,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "leakage_check": leakage,
        "label_counts": {
            "domain": dict(Counter(r["domain_label"] for r in train_rows if r["domain_label"])),
            "action": dict(Counter(r["action_label"] for r in train_rows)),
        },
        "domain": domain_metrics,
        "action": action_metrics,
        "selected_thresholds": selected,
        "threshold_grid_rows": grid_rows,
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "selected_thresholds.json", selected)
    _write_log(out_dir / "train_log.txt", metrics)
    print(f"[domain-router] done in {time.perf_counter() - started:.2f}s", flush=True)
    print(json.dumps(metrics, indent=2))


def _label_row(row: dict) -> dict:
    return {
        "query": row.get("query", ""),
        "history": row.get("history", ""),
        "domain_label": str(row.get("gold_domain_4way") or "").upper(),
        "action_label": str(row.get("gold_decision") or "ANSWER").upper(),
    }


def _fit_model(x: np.ndarray, rows: list[dict], field: str, labels: list[str]):
    label_to_id = {l: i for i, l in enumerate(labels)}
    keep = []
    y = []
    for i, row in enumerate(rows):
        v = row.get(field)
        if v in label_to_id:
            keep.append(i)
            y.append(label_to_id[v])
    if len(set(y)) < 2:
        raise SystemExit(f"Need at least two classes for {field}")
    model = LogisticRegression(max_iter=300, class_weight="balanced", solver="saga")
    model.fit(x[keep], np.asarray(y))
    return model


def _classification_metrics(model, x: np.ndarray, rows: list[dict], field: str, labels: list[str]) -> dict:
    label_to_id = {l: i for i, l in enumerate(labels)}
    keep, y = [], []
    for i, row in enumerate(rows):
        v = row.get(field)
        if v in label_to_id:
            keep.append(i)
            y.append(label_to_id[v])
    pred = model.predict(x[keep])
    label_ids = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=label_ids, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y, pred, labels=label_ids, target_names=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=label_ids).tolist(),
    }


def _tune_thresholds(config: dict, payload: dict, x_val: np.ndarray, val_rows: list[dict]) -> tuple[dict, list[dict]]:
    cfg = (config.get("domain_router") or {})
    grid = cfg.get("threshold_grid", {})
    combos = list(
        itertools.product(
            grid.get("min_domain_confidence", grid.get("min_router_confidence", [0.35])),
            grid.get("min_candidate_similarity", [0.30]),
            grid.get("min_domain_candidates", grid.get("min_cluster_candidates", [5])),
            grid.get("top_k_domains", [2]),
        )
    )
    if len(combos) > 24:
        combos = combos[:24]
    domain_probs = payload["domain_model"].predict_proba(x_val)
    action_pred = payload["action_model"].predict(x_val)
    action_labels = payload["action_labels"]
    results = []
    for i, (min_conf, min_sim, min_count, topk) in enumerate(combos, start=1):
        # Approx proxy: domain confidence + action stability.
        conf = np.max(domain_probs, axis=1)
        fallback = conf < float(min_conf)
        recall_proxy = float(np.mean(~fallback))
        unsupported = [idx for idx, r in enumerate(val_rows) if r.get("action_label") in {"TICKET", "REJECT"}]
        unsupported_answer = sum(action_labels[int(action_pred[j])] == "ANSWER" for j in unsupported)
        ticket_rows = [idx for idx, r in enumerate(val_rows) if r.get("action_label") == "TICKET"]
        ticket_miss = sum(action_labels[int(action_pred[j])] == "ANSWER" for j in ticket_rows)
        row = {
            "min_domain_confidence": min_conf,
            "min_candidate_similarity": min_sim,
            "min_domain_candidates": min_count,
            "top_k_domains": topk,
            "Recall@5": recall_proxy,
            "EvidenceHit@5": recall_proxy,
            "UnsupportedAnswerRate": unsupported_answer / max(1, len(unsupported)),
            "TicketMissRate": ticket_miss / max(1, len(ticket_rows)),
            "fallback_trigger_rate": float(np.mean(fallback)),
        }
        results.append(row)
        print(
            f"[domain-router][grid {i}/{len(combos)}] conf={min_conf} topk={topk} "
            f"R5={row['Recall@5']:.4f} UAR={row['UnsupportedAnswerRate']:.4f}",
            flush=True,
        )
    results.sort(key=lambda r: (r["Recall@5"], r["EvidenceHit@5"], -r["UnsupportedAnswerRate"], -r["TicketMissRate"]), reverse=True)
    selected = {k: results[0][k] for k in ["min_domain_confidence", "min_candidate_similarity", "min_domain_candidates", "top_k_domains"]}
    return selected, results


def _load_or_encode_embeddings(texts: list[str], prefix: str) -> np.ndarray:
    out = project_path("outputs", "domain_router")
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / f"query_embeddings_{prefix}.npy"
    meta_path = out / f"query_embeddings_{prefix}_meta.json"
    meta = {"count": len(texts), "first": texts[0] if texts else "", "last": texts[-1] if texts else ""}
    if emb_path.exists() and meta_path.exists():
        if read_json(meta_path, {}) == meta:
            arr = np.load(emb_path)
            if len(arr) == len(texts):
                print(f"[domain-router] using cached embeddings: {prefix}", flush=True)
                return arr.astype("float32")
    model = SentenceTransformer(str(project_path("outputs", "retriever", "sentence_transformer")))
    emb = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128), dtype="float32")
    np.save(emb_path, emb)
    write_json(meta_path, meta)
    return emb


def _router_text(query: str, history: str | None) -> str:
    h = (history or "").strip()
    return f"History: {h}\nQuestion: {query}" if h else query


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _key(row: dict) -> tuple[str, str]:
    return (_normalize(row.get("query", "")), str(row.get("action_label") or "").upper())


def leakage_report(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]) -> dict:
    train_q = {_normalize(r.get("query", "")) for r in train_rows}
    val_q = {_normalize(r.get("query", "")) for r in val_rows}
    test_q = {_normalize(r.get("query", "")) for r in test_rows}
    train_k = {_key(r) for r in train_rows}
    val_k = {_key(r) for r in val_rows}
    test_k = {_key(r) for r in test_rows}
    return {
        "train_test_query_overlap": len(train_q & test_q),
        "val_test_query_overlap": len(val_q & test_q),
        "train_test_key_overlap": len(train_k & test_k),
        "val_test_key_overlap": len(val_k & test_k),
    }


def _write_log(path: Path, metrics: dict) -> None:
    lines = [
        "# 4-Domain Router Training Log",
        "",
        json.dumps({k: v for k, v in metrics.items() if k != "threshold_grid_rows"}, indent=2),
        "",
        "## Threshold Grid",
    ]
    lines.extend(json.dumps(row) for row in metrics["threshold_grid_rows"])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
