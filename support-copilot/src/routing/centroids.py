from __future__ import annotations

from collections import defaultdict

from src.retrieval.search_kb import embed_text, normalize


def build_domain_centroids(chunks: list[dict], keywords: dict[str, list[str]] | None = None) -> list[dict]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["domain"]].append(embed_text(chunk["text"]))
    rows = []
    for domain, vectors in grouped.items():
        dim = len(vectors[0])
        mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
        rows.append(
            {
                "domain": domain,
                "centroid": normalize(mean),
                "num_chunks": len(vectors),
                "top_keywords": (keywords or {}).get(domain, [])[:30],
            }
        )
    return sorted(rows, key=lambda r: r["domain"])

