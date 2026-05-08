from __future__ import annotations

import re
import string
from typing import Any


CITATION_RE = re.compile(r"\[doc_id=(.*?), chunk_id=(.*?), span=(\d+)-(\d+)\]")
CONTINUATION_WORDS = {"and", "or", "but", "than", "to", "from", "with", "because", "that"}
SUPPORT_TERMS = {
    "benefit",
    "benefits",
    "ssi",
    "social security",
    "dmv",
    "license",
    "registration",
    "vehicle",
    "va",
    "veteran",
    "student aid",
    "fafsa",
    "loan",
    "claim",
    "account",
    "case",
    "renew",
    "renewal",
    "portal",
}


def clean_answer_text(answer: str) -> str:
    cleaned, _ = strip_inline_citations(answer)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_inline_citations(answer: str) -> tuple[str, list[str]]:
    citations = [m.group(0) for m in CITATION_RE.finditer(answer or "")]
    return CITATION_RE.sub("", answer or "").strip(), citations


def citation_identity(citation: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(citation.get("doc_id", "")),
        str(citation.get("chunk_id", "")),
        int(citation.get("span_start", 0) or 0),
        int(citation.get("span_end", 0) or 0),
    )


def deduplicate_citations(citations: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for citation in citations or []:
        ident = citation_identity(citation)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(citation)
    return out


def extract_inline_citation_dicts(answer: str) -> list[dict[str, Any]]:
    out = []
    for match in CITATION_RE.finditer(answer or ""):
        out.append(
            {
                "doc_id": match.group(1),
                "chunk_id": match.group(2),
                "span_start": int(match.group(3)),
                "span_end": int(match.group(4)),
            }
        )
    return out


def is_fragment_answer(answer: str, min_answer_words: int = 6) -> bool:
    raw = (answer or "").strip()
    if re.search(r"\[\d+\]", raw):
        return True
    text = clean_answer_text(answer)
    words = re.findall(r"\b\w+\b", text)
    if len(words) < min_answer_words:
        return True
    if not text:
        return True
    if text[0] in string.punctuation:
        return True
    if words and words[0].lower() in CONTINUATION_WORDS:
        return True
    if text[-1:] in ",;:":
        return True
    if len(text) < 80 and text[-1:] not in ".!?":
        return True
    return False


def is_support_like(query: str) -> bool:
    q = f" {(query or '').lower()} "
    return any(term in q for term in SUPPORT_TERMS)


def is_vague_query(query: str, max_centroid: float = 0.0, nearest_kb_similarity: float = 0.0) -> bool:
    q = (query or "").strip().lower()
    normalized = re.sub(rf"[{re.escape(string.punctuation)}]", "", q)
    words = re.findall(r"\b\w+\b", normalized)
    vague_patterns = {
        "why am i here",
        "what is this",
        "can you help me",
        "tell me more",
        "what should i do",
    }
    if normalized in vague_patterns:
        return True
    weak_signals = max_centroid < 0.25 and nearest_kb_similarity < 0.30
    if len(words) <= 4 and weak_signals and not is_support_like(query):
        return True
    few_specifics = len([w for w in words if len(w) > 4]) <= 1
    return weak_signals and few_specifics and not is_support_like(query)


def validate_answer(answer: str, query: str, citations: list[dict], min_answer_words: int = 6) -> dict:
    cleaned = clean_answer_text(answer)
    deduped = deduplicate_citations(citations)
    invalid_reasons = []
    if is_fragment_answer(cleaned, min_answer_words):
        invalid_reasons.append("fragment_answer")
    if not deduped:
        invalid_reasons.append("missing_citation")
    if len(re.findall(r"\b\w+\b", cleaned)) < 3:
        invalid_reasons.append("empty_or_invalid")
    if is_vague_query(query):
        invalid_reasons.append("vague_query")
    return {
        "valid": not invalid_reasons,
        "answer": cleaned,
        "citations": deduped,
        "invalid_reasons": invalid_reasons,
        "support_like": is_support_like(query),
    }
