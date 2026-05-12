from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import numpy as np

from src.retrieval.search_kb import search
from src.utils.io import project_path, read_json

FOUR_TO_KB = {"SSA": "ssa", "VA": "va", "SA": "studentaid", "FSA": "dmv"}


def _cfg(config: dict) -> dict:
    base = config.get("domain_router") if isinstance(config.get("domain_router"), dict) else {}
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    return {**retrieval, **base}


def load_router(config: dict) -> dict[str, Any]:
    cfg = _cfg(config)
    path = project_path(*str(cfg.get("model_path", "outputs/domain_router/model.pkl")).split("/"))
    if not bool(cfg.get("enabled", False)) or not path.exists():
        return {"enabled": False, "reason": "disabled_or_missing_model"}
    with path.open("rb") as f:
        payload = pickle.load(f)
    selected = read_json(
        project_path(*str(cfg.get("selected_thresholds_path", "outputs/domain_router/selected_thresholds.json")).split("/")),
        {},
    )
    # Runtime config must override saved selected thresholds during sweeps.
    return {"enabled": True, "payload": payload, "thresholds": {**selected, **cfg}}


def route_query(query: str, history: str | None, config: dict) -> dict[str, Any]:
    router = load_router(config)
    if not router.get("enabled"):
        return {"enabled": False, "reason": router.get("reason", "disabled")}
    payload = router["payload"]
    emb = _embed(_router_text(query))
    domain = _predict(payload["domain_model"], payload["domain_labels"], emb)
    action = _predict(payload["action_model"], payload["action_labels"], emb)
    top_k = int(router["thresholds"].get("top_k_domains", 2))
    top_domains = domain["top_labels"][:top_k]
    return {
        "enabled": True,
        "query": query,
        "domain": domain,
        "action": action,
        "top_domains": top_domains,
        "kb_domains": [FOUR_TO_KB.get(d, "") for d in top_domains if FOUR_TO_KB.get(d)],
        "router_confidence": float(domain["confidence"]),
        "thresholds": router["thresholds"],
    }


def retrieve_primary_domains(query: str, routed: dict, config: dict) -> tuple[list[dict], dict]:
    thresholds = routed.get("thresholds") or _cfg(config)
    top_k = int(config.get("top_k_retrieval", 20))
    kb_domains = [d for d in routed.get("kb_domains", []) if d]
    hits: list[dict] = []
    for d in kb_domains:
        hits.extend(search(query, top_k=top_k, domain=d))
    hits = _merge_hits(hits)[:top_k]
    best_score = float(hits[0]["score"]) if hits else 0.0
    fallback_reasons = []
    min_domain_candidates = int(thresholds.get("min_domain_candidates", thresholds.get("min_cluster_candidates", 5)))
    min_domain_confidence = float(thresholds.get("min_domain_confidence", thresholds.get("min_router_confidence", 0.7)))
    if len(hits) < min_domain_candidates:
        fallback_reasons.append("too_few_domain_candidates")
    if best_score < float(thresholds.get("min_candidate_similarity", 0.30)):
        fallback_reasons.append("weak_domain_candidate_similarity")
    if float(routed.get("router_confidence", 0.0)) < min_domain_confidence:
        fallback_reasons.append("low_domain_confidence")
    return hits, {
        "fallback_triggered": bool(fallback_reasons) and bool(thresholds.get("enable_global_fallback", True)),
        "fallback_reasons": fallback_reasons,
        "max_candidate_similarity": best_score,
        "domain_candidate_count": len(hits),
        "kb_domains": kb_domains,
    }


def _predict(model: Any, labels: list[str], emb: np.ndarray) -> dict[str, Any]:
    x = emb.reshape(1, -1)
    probs = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
    if probs is None:
        pred = int(model.predict(x)[0])
        probs = np.zeros(len(labels), dtype="float32")
        probs[pred] = 1.0
    order = np.argsort(probs)[::-1]
    return {
        "label": labels[int(order[0])],
        "confidence": float(probs[int(order[0])]),
        "top_labels": [labels[int(i)] for i in order],
        "probabilities": {labels[int(i)]: float(probs[int(i)]) for i in order},
    }


def _merge_hits(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for hit in sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True):
        key = hit.get("chunk_id")
        if key in seen:
            continue
        out.append(hit)
        seen.add(key)
    return out


def _router_text(query: str) -> str:
    return query


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    path = project_path("outputs", "retriever", "sentence_transformer")
    return SentenceTransformer(str(path) if path.exists() else "sentence-transformers/all-MiniLM-L6-v2")


def _embed(text: str) -> np.ndarray:
    vec = _model().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(vec, dtype="float32")
