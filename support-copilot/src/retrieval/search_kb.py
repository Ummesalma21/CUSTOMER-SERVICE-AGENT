from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.policy.answerability import support_topic_overlap_bonus
from src.utils.io import project_path, read_json, read_jsonl, write_json


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def embed_text(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for tok, count in Counter(tokenize(text)).items():
        digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    return normalize(vec)


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


_ST_MODEL = None
_LOADED_INDEX = None
_RUNTIME_INDEX = None
_RUNTIME_INDEX_ID = None
_QUERY_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}


def _load_sentence_transformer(path: str | Path):
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": "cuda"} if _cuda_available() else {}
        _ST_MODEL = SentenceTransformer(str(path), **kwargs)
    return _ST_MODEL


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _real_retriever_path() -> Path | None:
    meta = read_json(project_path("outputs", "retriever", "model.json"), {})
    if meta.get("backend") != "sentence-transformers":
        return None
    checkpoint = meta.get("checkpoint")
    if not checkpoint:
        return None
    path = project_path(*str(checkpoint).split("/"))
    return path if path.exists() else None


def build_index(chunks: list[dict] | None = None) -> dict:
    chunks = chunks or read_jsonl(project_path("data", "processed", "kb_chunks.jsonl"))
    st_path = _real_retriever_path()
    if st_path:
        model = _load_sentence_transformer(st_path)
        texts = [c["text"] for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist() if texts else []
        index = {
            "dim": len(vectors[0]) if vectors else 0,
            "chunks": [{**c, "embedding": v} for c, v in zip(chunks, vectors)],
            "backend": "sentence-transformers-json",
            "model_path": str(st_path.relative_to(project_path())),
        }
        write_json(project_path("data", "indexes", "kb_index.json"), index)
        _write_faiss(index)
        return index
    index = {
        "dim": 128,
        "chunks": [{**c, "embedding": embed_text(c["text"])} for c in chunks],
        "backend": "hashing-json",
    }
    write_json(project_path("data", "indexes", "kb_index.json"), index)
    return index


def load_index() -> dict:
    global _LOADED_INDEX
    if _LOADED_INDEX is not None:
        return _LOADED_INDEX
    index = read_json(project_path("data", "indexes", "kb_index.json"))
    if not index:
        index = build_index()
    _LOADED_INDEX = index
    return _LOADED_INDEX


def search(query: str, top_k: int = 5, domain: str | None = None, index: dict | None = None) -> list[dict]:
    index = index or load_index()
    q_tokens = set(tokenize(query))
    if str(index.get("backend", "")).startswith("sentence-transformers"):
        model_path = project_path(*str(index.get("model_path", "outputs/retriever/sentence_transformer")).split("/"))
        qvec = _query_embedding(query, model_path)
    else:
        qvec = embed_text(query, int(index.get("dim", 128)))
    fast_hits = _fast_search(index, qvec, q_tokens, top_k, domain, query)
    if fast_hits is not None:
        return fast_hits
    hits: list[dict] = []
    for chunk in index.get("chunks", []):
        if domain and chunk.get("domain") != domain:
            continue
        score = cosine(qvec, chunk["embedding"])
        chunk_text = " ".join(str(chunk.get(k, "")) for k in ("doc_id", "title", "text"))
        score += support_topic_overlap_bonus(query, chunk_text, max_bonus=0.75)
        item = {k: v for k, v in chunk.items() if k != "embedding"}
        item["score"] = round(float(score), 6)
        hits.append(item)
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:top_k]


def _query_embedding(query: str, model_path: Path) -> list[float]:
    key = (str(model_path), query)
    cached = _QUERY_EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    vec = _load_sentence_transformer(model_path).encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    if len(_QUERY_EMBED_CACHE) > 5000:
        _QUERY_EMBED_CACHE.clear()
    _QUERY_EMBED_CACHE[key] = vec
    return vec


def _fast_search(index: dict, qvec: list[float], q_tokens: set[str], top_k: int, domain: str | None, query: str) -> list[dict] | None:
    try:
        import numpy as np
    except Exception:
        return None
    chunks = index.get("chunks", [])
    if not chunks:
        return []
    runtime = _runtime_index(index)
    if runtime is None:
        return None
    vectors = runtime["vectors"]
    if vectors.size == 0:
        return []
    scores = vectors @ np.asarray(qvec, dtype="float32")
    candidate_mask = None
    if domain:
        candidate_mask = runtime["domains"] == domain
        if not bool(candidate_mask.any()):
            return []
        scores = scores.copy()
        scores[~candidate_mask] = -1e9
    overlap_bonus = _support_overlap_bonuses(query, runtime["payloads"])
    if overlap_bonus is not None:
        scores = scores.copy()
        scores += overlap_bonus
    limit = min(max(int(top_k), 0), len(chunks))
    if limit == 0:
        return []
    if candidate_mask is not None:
        limit = min(limit, int(candidate_mask.sum()))
    if limit <= 0:
        return []
    if limit < len(scores):
        top_idx = np.argpartition(scores, -limit)[-limit:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    else:
        top_idx = np.argsort(scores)[::-1]
    hits: list[dict] = []
    for idx in top_idx[:limit]:
        if scores[idx] <= -1e8:
            continue
        item = dict(runtime["payloads"][int(idx)])
        item["score"] = round(float(scores[idx]), 6)
        hits.append(item)
    return hits


def _runtime_index(index: dict) -> dict | None:
    global _RUNTIME_INDEX, _RUNTIME_INDEX_ID
    if _RUNTIME_INDEX is not None and _RUNTIME_INDEX_ID == id(index):
        return _RUNTIME_INDEX
    try:
        import numpy as np
    except Exception:
        return None
    chunks = index.get("chunks", [])
    vectors = np.asarray([c["embedding"] for c in chunks], dtype="float32")
    domains = np.asarray([str(c.get("domain", "")) for c in chunks])
    payloads = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
    _RUNTIME_INDEX = {
        "vectors": vectors,
        "domains": domains,
        "payloads": payloads,
    }
    _RUNTIME_INDEX_ID = id(index)
    return _RUNTIME_INDEX


def _support_overlap_bonuses(query: str, payloads: list[dict]):
    try:
        import numpy as np
    except Exception:
        return None
    values = [
        support_topic_overlap_bonus(query, " ".join(str(p.get(k, "")) for k in ("doc_id", "title", "text")), max_bonus=0.75)
        for p in payloads
    ]
    if not any(values):
        return None
    return np.asarray(values, dtype="float32")


def best_chunk_for_query(query: str, chunks: Iterable[dict], domain: str | None = None) -> dict | None:
    tmp = {"dim": 128, "chunks": [{**c, "embedding": embed_text(c["text"])} for c in chunks]}
    hits = search(query, top_k=1, domain=domain, index=tmp)
    return hits[0] if hits else None


def timed_search(query: str, top_k: int = 5, domain: str | None = None, index: dict | None = None) -> tuple[list[dict], float]:
    start = time.perf_counter()
    hits = search(query, top_k=top_k, domain=domain, index=index)
    return hits, (time.perf_counter() - start) * 1000.0


def _write_faiss(index: dict) -> None:
    try:
        import faiss
        import numpy as np

        vectors = np.array([c["embedding"] for c in index.get("chunks", [])], dtype="float32")
        if vectors.size == 0:
            return
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        out = project_path("data", "indexes", "kb_index.faiss")
        out.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, str(out))
    except Exception:
        return
