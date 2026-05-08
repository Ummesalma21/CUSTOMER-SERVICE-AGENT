from __future__ import annotations

from src.retrieval.search_kb import search
from src.routing.router import route_query
from src.utils.io import project_path, read_jsonl


def route_domain(query: str, top_k_domains: int = 2) -> dict:
    return route_query(query, top_k_domains)


def search_kb(query: str, top_k: int = 5, domain: str | None = None) -> dict:
    return {"passages": search(query, top_k=top_k, domain=domain)}


def get_policy(doc_id: str, section_id: str) -> dict:
    chunks = read_jsonl(project_path("data", "processed", "kb_chunks.jsonl"))
    text = " ".join(c["text"] for c in chunks if c["doc_id"] == doc_id and c["section_id"] == section_id)
    return {"policy_text": text, "doc_id": doc_id, "section_id": section_id}


def create_ticket(summary: str, category: str, severity: str = "medium") -> dict:
    ticket_id = "TCK-" + str(abs(hash((summary, category))) % 1000000).zfill(6)
    return {"ticket_id": ticket_id, "status": "created"}


def reject_query(reason: str, nearest_kb_distance: float, nearest_centroid_distance: float, confidence: float) -> dict:
    return {
        "decision": "rejected",
        "message": "I can only help with questions covered by this support knowledge base. Your question appears outside the supported domains, so I cannot answer it here.",
        "reason": reason,
        "nearest_kb_distance": nearest_kb_distance,
        "nearest_centroid_distance": nearest_centroid_distance,
        "confidence": confidence,
    }

