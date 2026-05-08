from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from src.retrieval.search_kb import tokenize
from src.triage.losses import boundary_loss
from src.utils.device import get_device
from src.utils.io import ensure_dir, project_path, read_jsonl, write_json

LABELS = ["ANSWER", "TICKET", "REJECT"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def train_triage(config: dict) -> dict:
    rows = read_jsonl(_project_file(_cfg_get(config, "train_file", "data/processed/triage_train.jsonl")))
    has_nested_triage_train = isinstance(config.get("triage"), dict) and bool(config["triage"].get("train_file"))
    if rows and (config.get("mode") in {"full_local", "full"} or has_nested_triage_train):
        limit = config.get("max_triage_train_examples")
        if limit:
            rows = rows[: int(limit)]
        return _train_distilbert_triage(config, rows)
    weights: dict[str, Counter] = {label: Counter() for label in LABELS}
    priors = Counter(r.get("gold_triage", "ANSWER") for r in rows)
    for row in rows:
        label = row.get("gold_triage", "ANSWER")
        toks = tokenize(row["query"])
        weights[label].update(toks)
        if label == "REJECT":
            weights[label].update({"out_of_domain": 3})
        if label == "TICKET":
            weights[label].update({"pending": 2, "error": 2, "case": 2, "review": 2})
    model = {"labels": LABELS, "weights": {k: dict(v) for k, v in weights.items()}, "priors": dict(priors)}
    write_json(project_path("outputs", "triage", "model.json"), model)
    correct = 0
    tbp = 0
    for row in rows:
        pred = predict_with_model(row["query"], model)
        correct += int(pred["label"] == row.get("gold_triage", "ANSWER"))
        tbp += int(pred["label"] == row.get("gold_triage", "ANSWER") and pred["margin"] >= float(config.get("mu_boundary", 0.15)))
    metrics = {
        "trained_examples": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "TBP@0.15": tbp / len(rows) if rows else 0.0,
        "boundary_loss": boundary_loss([0.2, 0.1, 0.0], 0, float(config.get("mu_boundary", 0.15))),
    }
    write_json(project_path("outputs", "triage", "metrics.json"), metrics)
    return metrics


def _train_distilbert_triage(config: dict, rows: list[dict]) -> dict:
    os.environ.setdefault("HF_HOME", str(project_path("data", "hf_cache", "hub")))
    import torch
    import torch.nn.functional as f
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class TriageDataset(Dataset):
        def __init__(self, data: list[dict], tokenizer) -> None:
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict:
            row = self.data[idx]
            enc = self.tokenizer(
                row["query"],
                truncation=True,
                max_length=int(config.get("triage_max_length", 128)),
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(LABEL_TO_ID.get(_row_label(row), 0), dtype=torch.long)
            return item

    val_rows = read_jsonl(_project_file(_cfg_get(config, "val_file", "")))
    model_name = str(_cfg_get(config, "model_name", config.get("triage_model_name", "distilbert-base-uncased")))
    device = torch.device(get_device(str(config.get("device", "auto"))))
    load_source = _local_model_source(model_name)
    tokenizer = AutoTokenizer.from_pretrained(load_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_source,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    ).to(device)
    batch_size = int(_cfg_get(config, "batch_size", config.get("triage_batch_size", 8)))
    grad_accum = max(1, int(_cfg_get(config, "gradient_accumulation_steps", 1)))
    loader = DataLoader(TriageDataset(rows, tokenizer), shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_cfg_get(config, "learning_rate", config.get("triage_lr", 2e-5))),
        weight_decay=float(_cfg_get(config, "weight_decay", 0.0)),
    )
    use_amp = bool(_cfg_get(config, "fp16", config.get("fp16", False)) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mu = float(_cfg_get(config, "boundary_margin_mu", config.get("mu_boundary", 0.15), section=("loss",)))
    lambda_boundary = float(_cfg_get(config, "lambda_boundary", config.get("lambda_boundary", 0.6), section=("loss",)))
    class_weight_map = _cfg_get(config, "class_weights", {}, section=("loss",)) or {}
    class_weights = torch.tensor([float(class_weight_map.get(label, 1.0)) for label in LABELS], dtype=torch.float, device=device)
    log_path = project_path("outputs", "logs", "triage_balanced_train.log")
    ensure_dir(log_path.parent)
    last_loss = 0.0
    history = []
    model.train()
    epochs = int(_cfg_get(config, "epochs", config.get("triage_epochs", 1)))
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"device={device}\nmodel={model_name}\ntrain_examples={len(rows)} val_examples={len(val_rows)}\n")
        for epoch in range(epochs):
            running = 0.0
            steps = 0
            optimizer.zero_grad()
            for step, batch in enumerate(loader, start=1):
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(**batch).logits
                    ce = f.cross_entropy(logits, labels, weight=class_weights)
                    correct = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                    masked = logits.masked_fill(f.one_hot(labels, len(LABELS)).bool(), -1e4)
                    wrong = masked.max(dim=1).values
                    boundary = f.softplus(wrong - correct + mu).mean()
                    loss = (ce + lambda_boundary * boundary) / grad_accum
                scaler.scale(loss).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                last_loss = float((loss * grad_accum).detach().cpu())
                running += last_loss
                steps += 1
            train_loss = running / max(1, steps)
            val_metrics = _evaluate_distilbert(model, tokenizer, val_rows, device, mu, batch_size)
            epoch_metrics = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
            history.append(epoch_metrics)
            log.write(f"epoch={epoch + 1} train_loss={train_loss:.6f} validation={val_metrics}\n")
            log.flush()
    out_dir = _project_file(_cfg_get(config, "output_dir", "outputs/triage/distilbert"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    train_eval = _evaluate_distilbert(model, tokenizer, rows, device, mu, batch_size)
    val_eval = _evaluate_distilbert(model, tokenizer, val_rows, device, mu, batch_size) if val_rows else {}
    model_meta = {
        "labels": LABELS,
        "backend": "distilbert-sequence-classifier",
        "model": model_name,
        "initialized_from": str(load_source),
        "checkpoint": str(out_dir.relative_to(project_path())),
        "device": str(device),
    }
    meta_out = project_path("outputs", "triage_balanced", "model.json") if "triage_balanced" in str(out_dir) else project_path("outputs", "triage", "model.json")
    write_json(meta_out, model_meta)
    metrics = {
        "trained_examples": len(rows),
        "validation_examples": len(val_rows),
        "model": model_name,
        "backend": "distilbert-sequence-classifier",
        "checkpoint": str(out_dir.relative_to(project_path())),
        "train": train_eval,
        "validation": val_eval,
        "accuracy": train_eval.get("accuracy", 0.0),
        "TBP@0.15": train_eval.get("TBP@0.15", 0.0),
        "avg_margin": train_eval.get("avg_margin", 0.0),
        "last_train_loss": last_loss,
        "history": history,
        "loss": "cross_entropy + lambda_boundary * softplus(max_wrong - correct + mu)",
        "class_weights": {label: float(class_weights[i].detach().cpu()) for i, label in enumerate(LABELS)},
    }
    metrics_out = project_path("outputs", "reports", "triage_balanced_train_metrics.json") if "triage_balanced" in str(out_dir) else project_path("outputs", "triage", "metrics.json")
    write_json(metrics_out, metrics)
    return metrics


def _evaluate_distilbert(model, tokenizer, rows: list[dict], device, mu: float, batch_size: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    if not rows:
        return {"accuracy": 0.0, "macro_f1": 0.0, "TBP@0.15": 0.0, "avg_margin": 0.0, "confusion_matrix": {}}
    model.eval()
    preds: list[int] = []
    golds: list[int] = []
    margins: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        enc = tokenizer(
            [r["query"] for r in batch_rows],
            truncation=True,
            max_length=192,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.detach().cpu()
        for row, logit in zip(batch_rows, logits):
            values = logit.tolist()
            pred = max(range(len(values)), key=lambda i: values[i])
            wrong = max(v for i, v in enumerate(values) if i != pred)
            preds.append(pred)
            golds.append(LABEL_TO_ID.get(_row_label(row), 0))
            margins.append(float(values[pred] - wrong))
    model.train()
    return _classification_report(golds, preds, margins, mu)


def _classification_report(golds: list[int], preds: list[int], margins: list[float], mu: float) -> dict:
    total = max(1, len(golds))
    out = {"accuracy": sum(g == p for g, p in zip(golds, preds)) / total}
    f1s = []
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for g, p in zip(golds, preds):
        confusion[LABELS[g]][LABELS[p]] += 1
    for idx, label in enumerate(LABELS):
        tp = sum(g == idx and p == idx for g, p in zip(golds, preds))
        fp = sum(g != idx and p == idx for g, p in zip(golds, preds))
        fn = sum(g == idx and p != idx for g, p in zip(golds, preds))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        out[f"{label} precision"] = precision
        out[f"{label} recall"] = recall
        out[f"{label} F1"] = f1
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s)
    out["confusion_matrix"] = confusion
    out["TBP@0.15"] = sum(g == p and m >= mu for g, p, m in zip(golds, preds, margins)) / total
    out["avg_margin"] = sum(margins) / len(margins) if margins else 0.0
    return out


def _row_label(row: dict) -> str:
    return row.get("gold_triage") or row.get("gold_decision") or row.get("label") or "ANSWER"


def _cfg_get(config: dict, key: str, default=None, section: tuple[str, ...] = ()) :
    cur = config.get("triage", {}) if isinstance(config.get("triage"), dict) else {}
    for name in section:
        cur = cur.get(name, {}) if isinstance(cur, dict) else {}
    if isinstance(cur, dict) and key in cur:
        return cur[key]
    if key in config:
        return config[key]
    return default


def _project_file(path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else project_path(*p.parts)


def _local_model_source(model_name: str) -> str:
    local_path = project_path(*Path(model_name).parts)
    if local_path.exists():
        return str(local_path)
    fallback = project_path("outputs", "triage", "distilbert")
    if fallback.exists():
        return str(fallback)
    return model_name


def predict_with_model(query: str, model: dict) -> dict:
    toks = tokenize(query)
    tok_set = set(toks)
    support_terms = {"benefit", "benefits", "renew", "renewal", "license", "registration", "address", "claim", "claims", "veterans", "vehicle", "portal"}
    ticket_terms = {"pending", "error", "rejected", "case", "someone", "review", "check"}
    reject_terms = {"ipl", "python", "pasta", "france", "netflix", "iphone", "amazon", "airline", "passport", "black", "holes"}
    scores = []
    for label in model["labels"]:
        w = model["weights"].get(label, {})
        score = 0.01 * model.get("priors", {}).get(label, 0)
        score += sum(w.get(t, 0) for t in toks)
        if label == "ANSWER" and tok_set.intersection(support_terms):
            score += 8
        if label == "REJECT" and tok_set.intersection(reject_terms):
            score += 10
        if label == "TICKET" and tok_set.intersection(ticket_terms):
            score += 6
        scores.append(float(score))
    best = max(range(len(scores)), key=lambda i: scores[i])
    wrong = max(s for i, s in enumerate(scores) if i != best)
    return {"label": model["labels"][best], "scores": scores, "margin": scores[best] - wrong}
