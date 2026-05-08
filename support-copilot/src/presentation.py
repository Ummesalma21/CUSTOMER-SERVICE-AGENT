from __future__ import annotations

import json
from typing import Any

from src.generation.answer_quality import (
    clean_answer_text,
    deduplicate_citations,
    extract_inline_citation_dicts,
    is_fragment_answer,
    is_support_like,
    is_vague_query,
)


def presentation_result(result: dict[str, Any], query: str, config: dict | None = None) -> dict[str, Any]:
    config = config or {}
    answering = config.get("answer_quality", {}) if isinstance(config.get("answer_quality"), dict) else {}
    if not answering and isinstance(config.get("answering"), dict):
        answering = config.get("answering", {})
    min_words = int(answering.get("min_answer_words", config.get("min_answer_words", 6)))
    block_fragments = bool(answering.get("block_fragment_answers", config.get("block_fragment_answers", True)))
    reject_vague = bool(answering.get("reject_vague_queries", config.get("reject_vague_queries", True)))
    out = dict(result)
    citations = deduplicate_citations((result.get("citations") or []) + extract_inline_citation_dicts(result.get("answer", "")))
    out["citations"] = citations
    out["display_answer"] = clean_answer_text(result.get("answer", ""))
    if reject_vague and out.get("decision") != "REJECT" and is_vague_query(query):
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
        support_like = is_support_like(query)
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
