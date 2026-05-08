from __future__ import annotations

import re

from src.generation.answer_quality import CONTINUATION_WORDS, is_fragment_answer
from src.retrieval.search_kb import tokenize


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def synthesize_extractive_answer(query: str, evidence_passages: list[dict], min_words: int = 6) -> dict:
    candidates = []
    q_tokens = set(tokenize(query))
    for passage in evidence_passages or []:
        for sentence in _sentences(passage.get("text", "")):
            if _bad_sentence(sentence, min_words):
                continue
            s_tokens = set(tokenize(sentence))
            if not s_tokens:
                continue
            overlap = len(q_tokens.intersection(s_tokens))
            score = overlap / max(1, len(q_tokens)) + 0.05 * float(passage.get("score", 0.0))
            candidates.append((score, sentence.strip(), passage))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates or candidates[0][0] <= 0.05:
        return {"status": "insufficient_evidence", "answer": None, "used_evidence_ids": [], "model_name": "extractive_synthesizer"}
    _, sentence, passage = candidates[0]
    answer = sentence
    if not _direct_answer(sentence):
        answer = f"Based on the support article, {sentence[0].lower() + sentence[1:]}"
    if answer[-1:] not in ".!?":
        answer += "."
    return {
        "status": "ok",
        "answer": answer,
        "used_evidence_ids": [str(passage.get("chunk_id", ""))],
        "model_name": "extractive_synthesizer",
    }


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    parts = SENTENCE_RE.split(normalized)
    if len(parts) == 1 and normalized[-1:] not in ".!?":
        return []
    return [p.strip() for p in parts if p.strip()]


def _bad_sentence(sentence: str, min_words: int) -> bool:
    words = re.findall(r"\b\w+\b", sentence)
    if len(words) < min_words:
        return True
    if words[0].lower() in CONTINUATION_WORDS:
        return True
    if sentence[0] in ",;:)]}":
        return True
    if is_fragment_answer(sentence, min_words):
        return True
    return False


def _direct_answer(sentence: str) -> bool:
    lower = sentence.lower()
    return lower.startswith(("you can", "you must", "you should", "if you", "eligible", "most "))
