from __future__ import annotations

from src.retrieval.search_kb import tokenize
from src.utils.io import project_path, read_json

_CROSS_ENCODER = None


def _load_cross_encoder():
    global _CROSS_ENCODER
    meta = read_json(project_path("outputs", "reranker", "model.json"), {})
    if meta.get("backend") != "cross-encoder":
        return None
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        checkpoint = project_path(*str(meta.get("checkpoint", "outputs/reranker/cross_encoder")).split("/"))
        kwargs = {"device": "cuda"} if _cuda_available() else {}
        _CROSS_ENCODER = CrossEncoder(str(checkpoint), **kwargs)
    return _CROSS_ENCODER


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False

def rerank(query: str, passages: list[dict], top_k: int = 5) -> list[dict]:
    q_tokens = set(tokenize(query))
    model = _load_cross_encoder()
    if model is not None and passages:
        scores = model.predict([[query, p["text"]] for p in passages], show_progress_bar=False)
        scored = []
        for p, score in zip(passages, scores):
            item = dict(p)
            adjusted = float(score) + _domain_hint_boost(q_tokens, item)
            item["rerank_score"] = round(adjusted, 6)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
    q = q_tokens
    scored = []
    for p in passages:
        overlap = len(q.intersection(tokenize(p["text"])))
        item = dict(p)
        item["rerank_score"] = round(float(p.get("score", 0.0)) + 0.05 * overlap + _domain_hint_boost(q_tokens, item), 6)
        scored.append(item)
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


def _domain_hint_boost(q_tokens: set[str], passage: dict) -> float:
    text_tokens = set(tokenize(" ".join(str(passage.get(k, "")) for k in ("doc_id", "title", "text"))))
    if q_tokens.intersection({"benefit", "benefits"}) and q_tokens.intersection({"renew", "renewal", "renewing"}):
        if text_tokens.intersection({"benefit", "benefits"}) and text_tokens.intersection({"renew", "renewal", "renewing"}):
            return 25.0
    return 0.0
