from __future__ import annotations

from src.retrieval.search_kb import search
from src.routing.domain_keywords import lexical_gate
from src.routing.router import route_query
from src.utils.io import project_path, read_json


def triage_features(query: str) -> dict:
    keywords = read_json(project_path("data", "processed", "domain_keywords.json"), {})
    gate = lexical_gate(query, keywords)
    route = route_query(query, top_k_domains=2)
    domains = route.get("domains", [])
    hits = search(query, top_k=2, domain=domains[0]["domain"] if domains else None)
    top_sim = domains[0]["centroid_similarity"] if domains else 0.0
    second_sim = domains[1]["centroid_similarity"] if len(domains) > 1 else 0.0
    chunk_sim = hits[0]["score"] if hits else 0.0
    gap = chunk_sim - (hits[1]["score"] if len(hits) > 1 else 0.0)
    return {
        "keyword_gate": "pass" if gate["pass"] else "reject",
        "nearest_centroid_domain": domains[0]["domain"] if domains else "none",
        "centroid_sim_top1": top_sim,
        "centroid_margin": top_sim - second_sim,
        "nearest_chunk_sim": chunk_sim,
        "retrieval_score_gap": gap,
        "matched_keywords": gate["match_count"],
    }


def feature_text(query: str) -> str:
    f = triage_features(query)
    return (
        f"query: {query}\nkeyword_gate: {f['keyword_gate']}\nnearest_centroid_domain: {f['nearest_centroid_domain']}\n"
        f"centroid_sim_top1: {f['centroid_sim_top1']:.3f}\ncentroid_margin: {f['centroid_margin']:.3f}\n"
        f"nearest_chunk_sim: {f['nearest_chunk_sim']:.3f}\nretrieval_score_gap: {f['retrieval_score_gap']:.3f}"
    )

