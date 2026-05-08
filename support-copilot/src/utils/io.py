from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        import yaml  # type: ignore

        with p.open("r", encoding="utf-8") as f:
            return normalize_config(yaml.safe_load(f) or {})
    except Exception:
        cfg: dict[str, Any] = {}
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            cfg[key.strip()] = _parse_scalar(value.strip())
        return normalize_config(cfg)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map nested experiment configs onto the original flat script keys."""
    out = dict(cfg)
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    training = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    routing = cfg.get("routing") if isinstance(cfg.get("routing"), dict) else {}
    triage = cfg.get("triage") if isinstance(cfg.get("triage"), dict) else {}
    class_balance = cfg.get("class_balance") if isinstance(cfg.get("class_balance"), dict) else {}

    mapping = {
        "max_eval_queries": "max_eval_samples",
        "max_retriever_train_pairs": "max_retriever_train_pairs",
        "max_reranker_train_pairs": "max_reranker_train_pairs",
        "max_triage_train_examples": "max_triage_train_examples",
        "max_preference_pairs": "max_preference_pairs",
        "max_kb_chunks": "max_kb_chunks",
    }
    for nested, flat in mapping.items():
        if nested in data and flat not in out:
            out[flat] = data[nested]
    if "use_all_kb_docs" in data and "max_kb_chunks" not in out:
        out["max_kb_chunks"] = None if data["use_all_kb_docs"] else out.get("max_kb_chunks")
    if data:
        out.setdefault("use_hf_dataset", True)
        out.setdefault("require_hf_dataset", True)
    for key, value in training.items():
        out.setdefault(key, value)
    routing_mapping = {
        "top_k_domains": "top_k_domains",
        "fallback_to_global_search": "fallback_to_global_search",
        "fallback_score_threshold": "fallback_score_threshold",
    }
    for nested, flat in routing_mapping.items():
        if nested in routing and flat not in out:
            out[flat] = routing[nested]
    triage_mapping = {
        "reject_threshold": "reject_threshold",
        "nearest_kb_similarity_threshold": "reject_nearest_kb_similarity_threshold",
        "reject_nearest_kb_similarity_threshold": "reject_nearest_kb_similarity_threshold",
        "centroid_similarity_threshold": "reject_centroid_similarity_threshold",
        "reject_centroid_similarity_threshold": "reject_centroid_similarity_threshold",
        "reject_require_lexical_low": "reject_require_lexical_low",
        "ticket_threshold": "ticket_threshold",
    }
    for nested, flat in triage_mapping.items():
        if nested in triage and flat not in out:
            out[flat] = triage[nested]
    if class_balance:
        out["class_balance"] = class_balance

    mode = out.get("mode")
    if mode == "full":
        out.setdefault("use_hf_dataset", True)
        out.setdefault("require_hf_dataset", True)
        out.setdefault("include_fixture_docs", True)
        out.setdefault("retriever_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        out.setdefault("reranker_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        out.setdefault("triage_model_name", "distilbert-base-uncased")
        out.setdefault("triage_max_length", 128)
        out.setdefault("triage_lr", 2e-5)
        out.setdefault("batch_size", 8)
        out.setdefault("reranker_batch_size", 4)
        out.setdefault("triage_batch_size", 8)
        out.setdefault("top_k_retrieval", 20)
        out.setdefault("top_k_rerank", 5)
        out.setdefault("top_k_domains", 2)
        out.setdefault("fallback_to_global_search", False)
        out.setdefault("fallback_score_threshold", 0.30)
        out.setdefault("reject_threshold", 0.18)
        out.setdefault("reject_nearest_kb_similarity_threshold", out.get("reject_threshold", 0.18))
        out.setdefault("reject_centroid_similarity_threshold", out.get("tau_domain", 0.35))
        out.setdefault("reject_require_lexical_low", True)
        out.setdefault("ticket_threshold", 0.20)
        out.setdefault("tau_domain", 0.35)
        out.setdefault("tau_chunk", 0.30)
        out.setdefault("mu_boundary", 0.15)
        out.setdefault("lambda_boundary", 0.6)
    return out


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")
