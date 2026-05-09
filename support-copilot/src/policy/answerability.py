from __future__ import annotations

import math
import re


SUPPORT_DOMAIN_TERMS = {
    "benefit",
    "benefits",
    "ssi",
    "social",
    "security",
    "dmv",
    "license",
    "registration",
    "vehicle",
    "veteran",
    "veterans",
    "student",
    "aid",
    "fafsa",
    "loan",
    "claim",
    "account",
    "case",
    "portal",
    "application",
}

SUPPORT_ACTION_TERMS = {
    "apply",
    "application",
    "check",
    "claim",
    "document",
    "documents",
    "eligibility",
    "eligible",
    "enroll",
    "payment",
    "renew",
    "renewal",
    "replace",
    "request",
    "submit",
    "update",
    "upload",
}

ACCOUNT_SPECIFIC_PATTERNS = (
    r"\bacct[- ]?\d+\b",
    r"\bcase[- #]?\d+\b",
    r"\bclaim[- #]?\d+\b",
    r"\btransaction[- #]?\d+\b",
    r"\bmy (account|case|claim|application|record|file|payment|portal)\b",
)

def tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]+\b", text or "")}


def support_keyword_score(query: str) -> float:
    tokens = tokenize(query)
    if not tokens:
        return 0.0
    domain_hits = len(tokens & SUPPORT_DOMAIN_TERMS)
    action_hits = len(tokens & SUPPORT_ACTION_TERMS)
    return min(1.0, 0.18 * domain_hits + 0.14 * action_hits)


def is_account_specific_query(query: str) -> bool:
    q = (query or "").lower()
    return any(re.search(pattern, q) for pattern in ACCOUNT_SPECIFIC_PATTERNS)


def is_vague_query(query: str, max_centroid: float = 0.0, nearest_kb_similarity: float = 0.0) -> bool:
    q = re.sub(r"[^a-zA-Z0-9\s]", "", (query or "").strip().lower())
    words = tokenize(q)
    weak_retrieval = max_centroid < 0.25 and nearest_kb_similarity < 0.30
    if support_keyword_score(query) > 0:
        return False
    vague_words = {"why", "what", "this", "here", "help", "tell", "more", "should", "do"}
    mostly_vague = bool(words) and len(words - vague_words) <= 1
    return weak_retrieval and (len(words) <= 4 or mostly_vague or len([w for w in words if len(w) > 4]) <= 1)


def lexical_similarity(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / math.sqrt(len(a_tokens) * len(b_tokens))


def has_strong_evidence(query: str, evidence: str, score: float, threshold: float = 0.30) -> bool:
    if score >= threshold and lexical_similarity(query, evidence) >= 0.12:
        return True
    return score >= threshold + 0.15 and support_keyword_score(query) > 0


def support_topic_overlap_bonus(query: str, evidence_text: str, max_bonus: float = 0.25) -> float:
    q_tokens = tokenize(query)
    e_tokens = tokenize(evidence_text)
    if not q_tokens or not e_tokens:
        return 0.0
    domain_overlap = len((q_tokens & SUPPORT_DOMAIN_TERMS) & e_tokens)
    action_overlap = len((q_tokens & SUPPORT_ACTION_TERMS) & e_tokens)
    if domain_overlap == 0 and action_overlap == 0:
        return 0.0
    if domain_overlap and action_overlap:
        return min(max_bonus, 0.45 + 0.08 * domain_overlap + 0.06 * action_overlap)
    return min(max_bonus, 0.08 * domain_overlap + 0.06 * action_overlap)


def should_reject(
    query: str,
    lexical_low: bool,
    centroid_similarity: float,
    nearest_kb_similarity: float,
    centroid_threshold: float,
    kb_threshold: float,
    require_lexical_low: bool = True,
) -> bool:
    lexical_ok = lexical_low if require_lexical_low else True
    return (
        lexical_ok
        and support_keyword_score(query) == 0.0
        and centroid_similarity < centroid_threshold
        and nearest_kb_similarity < kb_threshold
    )


def should_ticket(query: str, evidence_score: float, ticket_threshold: float = 0.20) -> bool:
    return is_account_specific_query(query) and support_keyword_score(query) > 0 and evidence_score < ticket_threshold


def should_answer(query: str, evidence_text: str, evidence_score: float, threshold: float = 0.30) -> bool:
    return not is_vague_query(query) and has_strong_evidence(query, evidence_text, evidence_score, threshold)
