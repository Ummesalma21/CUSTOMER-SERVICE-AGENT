from __future__ import annotations

import re
from typing import Any


def chunk_documents(documents: list[dict[str, Any]], max_words: int = 80) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc in documents:
        words = re.findall(r"\S+", doc.get("text", ""))
        if not words:
            continue
        for idx in range(0, len(words), max_words):
            part = words[idx : idx + max_words]
            span_start = idx
            span_end = idx + len(part)
            chunk_id = f"{doc['doc_id']}_span{idx // max_words:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "domain": doc.get("domain", "unknown"),
                    "section_id": f"section_{idx // max_words}",
                    "span_start": span_start,
                    "span_end": span_end,
                    "title": doc.get("title", doc["doc_id"]),
                    "text": " ".join(part),
                }
            )
    return chunks

