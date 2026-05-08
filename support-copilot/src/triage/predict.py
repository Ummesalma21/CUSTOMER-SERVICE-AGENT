from __future__ import annotations

from src.triage.train_triage import predict_with_model
from src.utils.io import project_path, read_json

_TRIAGE = {}


def predict_triage(query: str, config: dict | None = None) -> dict:
    model = _model_meta(config or {})
    if not model:
        return {"label": "ANSWER", "scores": [1.0, 0.0, 0.0], "margin": 1.0}
    if model.get("backend") == "distilbert-sequence-classifier":
        return _predict_distilbert(query, model)
    return predict_with_model(query, model)


def _model_meta(config: dict) -> dict:
    checkpoint = config.get("triage_checkpoint") or config.get("triage_model_checkpoint")
    if not checkpoint and isinstance(config.get("triage"), dict):
        checkpoint = config["triage"].get("checkpoint") or config["triage"].get("model_checkpoint")
    if checkpoint:
        return {
            "labels": ["ANSWER", "TICKET", "REJECT"],
            "backend": "distilbert-sequence-classifier",
            "checkpoint": checkpoint,
        }
    meta_path = config.get("triage_model_meta")
    if meta_path:
        return read_json(project_path(*str(meta_path).split("/")), {})
    return read_json(project_path("outputs", "triage", "model.json"))


def _predict_distilbert(query: str, meta: dict) -> dict:
    global _TRIAGE
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = project_path(*str(meta.get("checkpoint", "outputs/triage/distilbert")).replace("\\", "/").split("/"))
    key = str(checkpoint)
    if key not in _TRIAGE:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        _TRIAGE[key] = (tokenizer, model)
    tokenizer, model = _TRIAGE[key]
    with torch.no_grad():
        enc = tokenizer(query, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(0).detach().cpu().tolist()
    labels = meta.get("labels", ["ANSWER", "TICKET", "REJECT"])
    best = max(range(len(logits)), key=lambda i: logits[i])
    wrong = max(v for i, v in enumerate(logits) if i != best)
    return {"label": labels[best], "scores": [float(v) for v in logits], "margin": float(logits[best] - wrong)}
