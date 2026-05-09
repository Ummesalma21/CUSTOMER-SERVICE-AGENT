from __future__ import annotations

import re
from typing import Any


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def chunk_documents(
    documents: list[dict[str, Any]],
    max_words: int = 90,
    min_words: int = 18,
    sentence_overlap: int = 1,
) -> list[dict[str, Any]]:
    """Build sentence-aware KB chunks instead of fixed word windows.

    The old chunker split every N words, which often produced chunks that began
    mid-sentence. That is acceptable for retrieval but poor evidence for a
    generator. This chunker keeps complete sentences together, uses paragraph
    boundaries as soft section boundaries, and carries a small sentence overlap
    for context.
    """
    chunks: list[dict[str, Any]] = []
    for doc in documents:
        text = _normalize_text(str(doc.get("text", "")))
        if not text:
            continue
        sections = _sections(text)
        chunk_idx = 0
        word_cursor = 0
        for section_idx, section in enumerate(sections):
            sentences = _sentences(section)
            if not sentences:
                continue
            pending: list[str] = []
            pending_start = word_cursor
            for sentence in sentences:
                sentence_words = _words(sentence)
                if not sentence_words:
                    continue
                pending_words = sum(len(_words(s)) for s in pending)
                if pending and pending_words + len(sentence_words) > max_words:
                    chunk_idx = _append_chunk(chunks, doc, pending, pending_start, section_idx, chunk_idx)
                    overlap = pending[-sentence_overlap:] if sentence_overlap > 0 else []
                    pending = list(overlap)
                    pending_start = max(word_cursor - sum(len(_words(s)) for s in pending), 0)
                if len(sentence_words) > max_words:
                    if pending:
                        chunk_idx = _append_chunk(chunks, doc, pending, pending_start, section_idx, chunk_idx)
                        pending = []
                    for part in _split_long_sentence(sentence, max_words):
                        chunk_idx = _append_chunk(chunks, doc, [part], word_cursor, section_idx, chunk_idx)
                        word_cursor += len(_words(part))
                    pending_start = word_cursor
                    continue
                if not pending:
                    pending_start = word_cursor
                pending.append(sentence)
                word_cursor += len(sentence_words)
            if pending:
                if chunks and sum(len(_words(s)) for s in pending) < min_words:
                    prev = chunks[-1]
                    merged = f"{prev['text']} {' '.join(pending)}"
                    if len(_words(merged)) <= max_words + 20 and prev["doc_id"] == doc["doc_id"]:
                        prev["text"] = merged
                        prev["span_end"] = prev["span_start"] + len(_words(merged))
                    else:
                        chunk_idx = _append_chunk(chunks, doc, pending, pending_start, section_idx, chunk_idx)
                else:
                    chunk_idx = _append_chunk(chunks, doc, pending, pending_start, section_idx, chunk_idx)
    return chunks


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sections(text: str) -> list[str]:
    sections = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    return sections or [text]


def _sentences(section: str) -> list[str]:
    rough = []
    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue
        rough.extend(SENTENCE_RE.split(line))
    sentences = []
    for item in rough:
        cleaned = item.strip(" -\t")
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    words = _words(sentence)
    parts = []
    for idx in range(0, len(words), max_words):
        part = " ".join(words[idx : idx + max_words]).strip()
        if part and part[-1] not in ".!?":
            part += "."
        parts.append(part)
    return parts


def _append_chunk(
    chunks: list[dict[str, Any]],
    doc: dict[str, Any],
    sentences: list[str],
    span_start: int,
    section_idx: int,
    chunk_idx: int,
) -> int:
    text = " ".join(s.strip() for s in sentences if s.strip()).strip()
    if not text:
        return chunk_idx
    if text[-1] not in ".!?":
        text += "."
    words = _words(text)
    chunk_id = f"{doc['doc_id']}_span{chunk_idx:04d}"
    chunks.append(
        {
            "chunk_id": chunk_id,
            "doc_id": doc["doc_id"],
            "domain": doc.get("domain", "unknown"),
            "section_id": f"section_{section_idx}",
            "span_start": span_start,
            "span_end": span_start + len(words),
            "title": doc.get("title", doc["doc_id"]),
            "text": text,
        }
    )
    return chunk_idx + 1


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")
