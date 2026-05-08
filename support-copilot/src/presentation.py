from __future__ import annotations

import json
import re
import string
from typing import Any


CITATION_RE = re.compile(r"\[doc_id=(.*?), chunk_id=(.*?), span=(\d+)-(\d+)\]")
CONTINUATION_WORDS = {"and", "or", "but", "than", "to", "from", "with", "because", "that"}
VAGUE_QUERIES = {
    "why am i here",
    "what is this",
    "can you help me",
    "tell me more",
    "what should i do",
}


def clean_inline_citations(text: str) -> str:
    cleaned = CITATION_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def citation_identity(citation: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(citation.get("doc_id", "")),
        str(citation.get("chunk_id", "")),
        int(citation.get("span_start", 0) or 0),
        int(citation.get("span_end", 0) or 0),
    )


def dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for citation in citations or []:
        ident = citation_identity(citation)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(citation)
    return out


def extract_inline_citations(text: str) -> list[dict[str, Any]]:
    out = []
    for match in CITATION_RE.finditer(text or ""):
        out.append(
            {
                "doc_id": match.group(1),
                "chunk_id": match.group(2),
                "span_start": int(match.group(3)),
                "span_end": int(match.group(4)),
            }
        )
    return out


def is_vague_query(query: str) -> bool:
    normalized = re.sub(rf"[{re.escape(string.punctuation)}]", "", (query or "").lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in VAGUE_QUERIES


def is_fragment_answer(answer: str, min_words: int = 6) -> bool:
    text = clean_inline_citations(answer)
    words = re.findall(r"\b\w+\b", text)
    if len(words) < min_words:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[0] in ",;:)]}":
        return True
    first = words[0].lower() if words else ""
    if first in CONTINUATION_WORDS:
        return True
    if len(stripped) < 80 and stripped[-1:] not in ".!?":
        return True
    return False


def presentation_result(result: dict[str, Any], query: str, config: dict | None = None) -> dict[str, Any]:
    config = config or {}
    answering = config.get("answering", {}) if isinstance(config.get("answering"), dict) else {}
    min_words = int(answering.get("min_answer_words", config.get("min_answer_words", 6)))
    block_fragments = bool(answering.get("block_fragment_answers", config.get("block_fragment_answers", True)))
    reject_vague = bool(answering.get("reject_vague_queries", config.get("reject_vague_queries", True)))
    out = dict(result)
    citations = dedupe_citations((result.get("citations") or []) + extract_inline_citations(result.get("answer", "")))
    out["citations"] = citations
    out["display_answer"] = clean_inline_citations(result.get("answer", ""))
    if reject_vague and is_vague_query(query):
        out["decision"] = "REJECT"
        out["display_answer"] = (
            "I need a more specific support question to search the knowledge base. Please ask about benefits, "
            "DMV services, VA benefits, or student aid policies."
        )
        out["answer"] = out["display_answer"]
        out["citations"] = []
        traces = list(out.get("tool_trace", []))
        traces.append(
            {
                "name": "RejectQuery",
                "arguments": {"reason": "underspecified_or_out_of_scope"},
                "returns": {"message": out["display_answer"]},
            }
        )
        out["tool_trace"] = traces
        return out
    if out.get("decision") == "ANSWER" and block_fragments and is_fragment_answer(out.get("answer", ""), min_words):
        support_like = _support_like(query)
        if support_like:
            out["decision"] = "TICKET"
            out["display_answer"] = "I found related support material, but not enough complete evidence to answer confidently. I created a support ticket for human review."
            out["answer"] = out["display_answer"]
            out["citations"] = []
            traces = list(out.get("tool_trace", []))
            if not any(t.get("name") == "CreateTicket" for t in traces):
                traces.append(
                    {
                        "name": "CreateTicket",
                        "arguments": {"summary": query, "category": "support", "severity": "medium"},
                        "returns": {"ticket_id": "presentation-guardrail"},
                    }
                )
            out["tool_trace"] = traces
        else:
            out["decision"] = "REJECT"
            out["display_answer"] = "I need a more specific support question to search the knowledge base. Please ask about benefits, DMV services, VA benefits, or student aid policies."
            out["answer"] = out["display_answer"]
            out["citations"] = []
    return out


def format_tool_trace(traces: list[dict[str, Any]]) -> str:
    if not traces:
        return "Tool trace: none"
    lines = ["Tool trace:"]
    for idx, call in enumerate(traces, start=1):
        lines.append(f"{idx}. {call.get('name', call.get('tool', 'Tool'))}")
        lines.append(f"   arguments: {json.dumps(call.get('arguments', {}), ensure_ascii=False)}")
    return "\n".join(lines)


def ticket_from_trace(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    for call in traces or []:
        if call.get("name") == "CreateTicket":
            args = call.get("arguments", {})
            returns = call.get("returns", {})
            return {**args, **returns}
    return None


def reject_from_trace(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    for call in traces or []:
        if call.get("name") == "RejectQuery":
            return {**call.get("arguments", {}), **call.get("returns", {})}
    return None


def _support_like(query: str) -> bool:
    q = (query or "").lower()
    terms = [
        "benefit",
        "benefits",
        "dmv",
        "license",
        "registration",
        "vehicle",
        "va",
        "veteran",
        "student aid",
        "fafsa",
        "loan",
        "account",
        "case",
        "claim",
        "pending",
        "manual review",
    ]
    return any(term in q for term in terms)
