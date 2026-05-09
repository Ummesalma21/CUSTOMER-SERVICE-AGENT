from __future__ import annotations

import json
import math
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.answer_quality import CONTINUATION_WORDS, clean_answer_text, is_fragment_answer
from src.generation.templates import cited_answer
from src.utils.io import project_path, read_jsonl, write_json, write_jsonl


PREDICTION_CANDIDATES = [
    "outputs/reports/final_answer_only_supported_synthesis_predictions.jsonl",
    "outputs/reports/final_answer_only_generator_fixed_predictions.jsonl",
    "outputs/reports/final_answer_only_generator_clean_predictions.jsonl",
    "outputs/reports/final_answer_only_predictions.jsonl",
    "outputs/reports/final_answer_quality_predictions_scored.jsonl",
]

DEFAULT_CONFIGS = [
    "configs/final_eval_generator_fixed.yaml",
    "configs/final_eval_balanced_triage_best.yaml",
]

THRESHOLDS = {
    "query_citation_similarity": 0.35,
    "answer_citation_similarity": 0.40,
    "query_answer_similarity": 0.30,
}

BROKEN_SPACING_PATTERNS = ("helpswith", "mostlong", "orskilled", "andmany", "seranyone")
PROCEDURE_TERMS = {
    "apply",
    "renew",
    "submit",
    "form",
    "online",
    "portal",
    "document",
    "documents",
    "process",
    "procedure",
    "step",
    "steps",
    "application",
}
INTENT_STOPWORDS = {
    "about",
    "after",
    "before",
    "could",
    "does",
    "have",
    "help",
    "here",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
    "can",
    "how",
    "the",
    "and",
    "for",
    "that",
    "this",
    "are",
    "you",
}


def main() -> None:
    pred_path = _find_or_create_predictions()
    rows = [r for r in read_jsonl(pred_path) if _is_answerable(r)]
    kb = _load_kb()
    embedder = _load_embedder()

    scored = []
    baseline_scores = []
    proposed_scores = []
    for row in rows:
        baseline = _system_payload(row, "baseline", kb)
        proposed = _system_payload(row, "proposed", kb)
        baseline_eval = _score_answer(row["query"], baseline["answer"], baseline["citations"], baseline["evidence_text"], embedder)
        proposed_eval = _score_answer(row["query"], proposed["answer"], proposed["citations"], proposed["evidence_text"], embedder)
        baseline_scores.append(baseline_eval)
        proposed_scores.append(proposed_eval)
        scored.append(
            {
                "query_id": row.get("query_id"),
                "query": row.get("query", ""),
                "gold_doc_id": row.get("gold_doc_id"),
                "gold_chunk_id": row.get("gold_chunk_id"),
                "reference_answer": row.get("reference_answer") or row.get("gold_answer") or "",
                "baseline": {**baseline, **baseline_eval},
                "proposed": {**proposed, **proposed_eval},
            }
        )

    metrics = {
        "prediction_file": str(pred_path),
        "evaluated_rows": len(rows),
        "thresholds": THRESHOLDS,
        "similarity_backend": "sentence_transformer" if embedder else "lexical_overlap_fallback",
        "baseline": _aggregate(baseline_scores),
        "proposed": _aggregate(proposed_scores),
    }
    write_json(project_path("outputs", "reports", "esa_aqs_metrics.json"), metrics)
    write_jsonl(project_path("outputs", "reports", "esa_aqs_scored_predictions.jsonl"), scored)
    _write_summary(metrics)
    print(json.dumps(metrics, indent=2))


def _find_or_create_predictions() -> Path:
    for rel in PREDICTION_CANDIDATES:
        path = project_path(*Path(rel).parts)
        if path.exists() and path.stat().st_size > 0:
            return path
    config = next((c for c in DEFAULT_CONFIGS if project_path(*Path(c).parts).exists()), DEFAULT_CONFIGS[-1])
    cmd = [
        sys.executable,
        "scripts/evaluate_answer_only.py",
        "--config",
        config,
        "--eval-path",
        "data/processed/eval_set.jsonl",
    ]
    subprocess.run(cmd, cwd=project_path(), check=True)
    path = project_path("outputs", "reports", "final_answer_only_predictions.jsonl")
    if not path.exists():
        raise FileNotFoundError("Could not find or create answer-only predictions.")
    return path


def _is_answerable(row: dict[str, Any]) -> bool:
    return bool(row.get("gold_chunk_id") or row.get("gold_doc_id") or row.get("reference_answer") or row.get("gold_answer"))


def _load_kb() -> dict[str, dict[str, Any]]:
    kb: dict[str, dict[str, Any]] = {}
    for rel in ["data/processed/kb_chunks.jsonl", "data/processed/kb.jsonl"]:
        path = project_path(*Path(rel).parts)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("chunk_id"):
                kb[str(row["chunk_id"])] = row
            key = f"{row.get('doc_id')}::{row.get('span_start')}-{row.get('span_end')}"
            kb[key] = row
    return kb


def _load_embedder():
    candidates = [
        project_path("outputs", "retriever", "sentence_transformer"),
        project_path("outputs", "retriever_fresh", "sentence_transformer"),
        project_path("outputs", "archive_runs", "20260508_225006", "outputs", "retriever", "sentence_transformer"),
    ]
    try:
        from sentence_transformers import SentenceTransformer

        for candidate in candidates:
            if candidate.exists():
                return SentenceTransformer(str(candidate))
    except Exception:
        return None
    return None


def _system_payload(row: dict[str, Any], prefix: str, kb: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hits = row.get(f"{prefix}_hits") or []
    citations = row.get(f"{prefix}_citations") or []
    if prefix == "baseline":
        citations = citations or hits[:1]
        answer = row.get("baseline_answer") or cited_answer(row.get("query", ""), hits)
    else:
        citations = citations or _inline_citations(row.get("proposed_answer", ""), kb)
        answer = row.get("proposed_answer") or row.get("answer") or ""
    evidence_text = _citation_text(citations, kb) or ((hits[0].get("text") or "") if hits else "")
    return {
        "decision": row.get(f"{prefix}_decision", "ANSWER" if prefix == "baseline" else ""),
        "answer": answer,
        "citations": citations,
        "evidence_text": evidence_text,
    }


def _inline_citations(answer: str, kb: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for match in re.finditer(r"\[doc_id=(.*?),\s*chunk_id=(.*?),\s*span=(.*?)\]", answer or ""):
        doc_id, chunk_id, span = match.groups()
        cit = {"doc_id": doc_id.strip(), "chunk_id": chunk_id.strip(), "span": span.strip()}
        if cit["chunk_id"] in kb:
            cit.update(kb[cit["chunk_id"]])
        citations.append(cit)
    return citations


def _citation_text(citations: list[dict[str, Any]], kb: dict[str, dict[str, Any]]) -> str:
    parts = []
    for citation in citations or []:
        if citation.get("text"):
            parts.append(str(citation["text"]))
            continue
        chunk_id = str(citation.get("chunk_id") or "")
        if chunk_id in kb:
            parts.append(str(kb[chunk_id].get("text", "")))
            continue
        key = f"{citation.get('doc_id')}::{citation.get('span_start')}-{citation.get('span_end')}"
        if key in kb:
            parts.append(str(kb[key].get("text", "")))
    return " ".join(p for p in parts if p).strip()


def _score_answer(query: str, answer: str, citations: list[dict[str, Any]], evidence_text: str, embedder) -> dict[str, Any]:
    cleaned = clean_answer_text(answer or "")
    malformed = _malformed(cleaned)
    sims = _similarities(query, cleaned, evidence_text, embedder)
    esa_pass = True
    reason = None
    if not citations:
        esa_pass, reason = False, "missing_citation"
    elif not evidence_text:
        esa_pass, reason = False, "citation_text_not_found"
    elif sims["query_citation_similarity"] < THRESHOLDS["query_citation_similarity"]:
        esa_pass, reason = False, "citation_not_relevant_to_query"
    elif sims["answer_citation_similarity"] < THRESHOLDS["answer_citation_similarity"]:
        esa_pass, reason = False, "answer_not_supported_by_citation"
    elif sims["query_answer_similarity"] < THRESHOLDS["query_answer_similarity"]:
        esa_pass, reason = False, "answer_not_direct_to_query"
    elif malformed:
        esa_pass, reason = False, "malformed_answer"

    fluency = _fluency_score(cleaned)
    correctness = _correctness_score(query, cleaned, sims["query_answer_similarity"], malformed)
    trueness = _trueness_score(bool(citations), sims, esa_pass)
    aqs = (fluency + correctness + trueness) / 6.0
    return {
        **sims,
        "esa_pass": bool(esa_pass),
        "esa_failure_reason": reason,
        "fluency_score": fluency,
        "correctness_score": correctness,
        "trueness_score": trueness,
        "aqs": aqs,
        "clean_answer": cleaned,
    }


def _similarities(query: str, answer: str, evidence: str, embedder) -> dict[str, float]:
    if embedder and query and answer and evidence:
        try:
            emb = embedder.encode([query, answer, evidence], normalize_embeddings=True)
            return {
                "query_citation_similarity": float(_dot(emb[0], emb[2])),
                "answer_citation_similarity": float(_dot(emb[1], emb[2])),
                "query_answer_similarity": float(_dot(emb[0], emb[1])),
            }
        except Exception:
            pass
    return {
        "query_citation_similarity": _lexical_similarity(query, evidence),
        "answer_citation_similarity": _lexical_similarity(answer, evidence),
        "query_answer_similarity": _lexical_similarity(query, answer),
    }


def _dot(a, b) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _lexical_similarity(a: str, b: str) -> float:
    ta = _important_tokens(a)
    tb = _important_tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    return overlap / math.sqrt(len(ta) * len(tb))


def _important_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower()) if t not in INTENT_STOPWORDS}


def _malformed(answer: str) -> bool:
    if not answer or is_fragment_answer(answer):
        return True
    stripped = answer.strip()
    if stripped[-1:] == "?":
        return True
    if stripped[0] in string.punctuation:
        return True
    first = re.findall(r"\b\w+\b", stripped[:40].lower())
    if first and first[0] in CONTINUATION_WORDS:
        return True
    lower = stripped.lower()
    if any(p in lower for p in BROKEN_SPACING_PATTERNS):
        return True
    if not re.search(r"[.!?]\s*$", stripped) and len(stripped) < 80:
        return True
    return False


def _fluency_score(answer: str) -> int:
    if _malformed(answer):
        return 0
    words = re.findall(r"\b\w+\b", answer)
    if len(words) < 10 or not re.search(r"[.!?]\s*$", answer.strip()):
        return 1
    return 2


def _correctness_score(query: str, answer: str, query_answer_similarity: float, malformed: bool) -> int:
    if malformed:
        return 0
    q = query.lower()
    a_tokens = _important_tokens(answer)
    if any(term in q for term in ["procedure", "process", "step", "steps", "how do", "how can", "renew", "apply"]):
        if not (a_tokens & PROCEDURE_TERMS):
            return min(1, 2 if query_answer_similarity >= 0.40 else 0)
    if any(term in q for term in ["eligible", "eligibility", "status", "account", "case"]) and re.search(r"\byou are eligible\b|\byou qualify\b", answer.lower()):
        if query_answer_similarity < 0.45:
            return 1
    if query_answer_similarity >= 0.40:
        return 2
    if query_answer_similarity >= 0.25:
        return 1
    return 0


def _trueness_score(has_citation: bool, sims: dict[str, float], esa_pass: bool) -> int:
    if esa_pass:
        return 2
    if has_citation and (
        sims["query_citation_similarity"] >= THRESHOLDS["query_citation_similarity"]
        or sims["answer_citation_similarity"] >= THRESHOLDS["answer_citation_similarity"]
    ):
        return 1
    return 0


def _aggregate(scores: list[dict[str, Any]]) -> dict[str, float]:
    denom = max(1, len(scores))
    return {
        "ESA": sum(1 for s in scores if s["esa_pass"]) / denom,
        "AQS": sum(float(s["aqs"]) for s in scores) / denom,
    }


def _write_summary(metrics: dict[str, Any]) -> None:
    b = metrics["baseline"]
    p = metrics["proposed"]
    improved = p["ESA"] > b["ESA"] and p["AQS"] > b["AQS"]
    if improved:
        interpretation = (
            f"The proposed system improves ESA from {b['ESA']:.4f} to {p['ESA']:.4f} and AQS from "
            f"{b['AQS']:.4f} to {p['AQS']:.4f} under the same automatic thresholds."
        )
    else:
        interpretation = (
            f"Proposed ESA/AQS did not improve on both metrics; ESA is {b['ESA']:.4f} vs {p['ESA']:.4f}, "
            f"and AQS is {b['AQS']:.4f} vs {p['AQS']:.4f}. This remains a limitation."
        )
    lines = [
        "# ESA and AQS Evaluation",
        "",
        f"Prediction file: `{metrics['prediction_file']}`",
        f"Evaluated answerable rows: `{metrics['evaluated_rows']}`",
        f"Similarity backend: `{metrics['similarity_backend']}`",
        "",
        "Thresholds:",
        "",
        f"- query_citation_similarity >= `{THRESHOLDS['query_citation_similarity']}`",
        f"- answer_citation_similarity >= `{THRESHOLDS['answer_citation_similarity']}`",
        f"- query_answer_similarity >= `{THRESHOLDS['query_answer_similarity']}`",
        "",
        "| Metric | Baseline RAG | Proposed |",
        "|---|---:|---:|",
        f"| ESA | {b['ESA']:.4f} | {p['ESA']:.4f} |",
        f"| AQS | {b['AQS']:.4f} | {p['AQS']:.4f} |",
        "",
        interpretation,
        "",
        "ESA is a binary automatic proxy for whether the final answer is supported by its cited evidence. "
        "AQS is a 0-to-1 automatic rubric averaging fluency, correctness/directness, and trueness/grounding. "
        "These are not human-evaluation scores.",
    ]
    project_path("outputs", "reports", "esa_aqs_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
