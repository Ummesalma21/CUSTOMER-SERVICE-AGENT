from __future__ import annotations


def cited_answer(query: str, passages: list[dict]) -> str:
    if not passages:
        return "I could not find enough KB evidence to answer this confidently."
    p = passages[0]
    sentence = p["text"].split(".")[0].strip()
    citation = f"[doc_id={p['doc_id']}, chunk_id={p['chunk_id']}, span={p['span_start']}-{p['span_end']}]"
    return f"{sentence}. {citation}"


def ticket_answer(ticket: dict, category: str) -> str:
    return (
        "I could not find enough KB evidence to answer this confidently, but your issue appears related to "
        f"{category}. I created a support ticket for human review.\nTicket ID: {ticket['ticket_id']}"
    )


def reject_answer() -> str:
    return "I can only help with questions covered by this support knowledge base. Your question appears outside the supported domains, so I cannot answer it here."

