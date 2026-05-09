from __future__ import annotations

from src.policy.answerability import support_topic_overlap_bonus
from src.retrieval.search_kb import cosine, embed_text, load_index
from src.routing.domain_keywords import lexical_gate
from src.utils.io import project_path, read_json


def route_query(query: str, top_k_domains: int = 2) -> dict:
    centroids = read_json(project_path("data", "processed", "domain_centroids.json"), [])
    keywords = read_json(project_path("data", "processed", "domain_keywords.json"), {})
    qvec = embed_text(query)
    routed = []
    gate = lexical_gate(query, keywords)
    for row in centroids:
        sim = cosine(qvec, row["centroid"])
        matched = gate["matches"].get(row["domain"], [])
        score = sim + 0.08 * len(matched)
        score += support_topic_overlap_bonus(query, " ".join(matched), max_bonus=0.20)
        routed.append(
            {
                "domain": row["domain"],
                "centroid_similarity": round(sim, 6),
                "route_score": round(score, 6),
                "centroid_distance": round(1.0 - sim, 6),
                "matched_keywords": matched,
                "num_chunks": row.get("num_chunks", 0),
            }
        )
    routed.sort(key=lambda r: r["route_score"], reverse=True)
    top = routed[:top_k_domains]
    confidence = top[0]["centroid_similarity"] - (top[1]["centroid_similarity"] if len(top) > 1 else 0.0) if top else 0.0
    return {"domains": top, "route_confidence": round(confidence, 6), "lexical_gate": gate}


def fraction_scanned(domains: list[str] | None = None) -> float:
    index = load_index()
    chunks = index.get("chunks", [])
    if not chunks:
        return 1.0
    if not domains:
        return 1.0
    count = sum(1 for c in chunks if c.get("domain") in set(domains))
    return count / len(chunks)
