from __future__ import annotations

import math
from collections import Counter, defaultdict

from src.retrieval.search_kb import tokenize


STOP = {
    "the",
    "and",
    "you",
    "can",
    "for",
    "with",
    "online",
    "must",
    "may",
    "this",
    "that",
    "your",
    "are",
    "how",
    "what",
}


def build_domain_keywords(chunks: list[dict], top_n: int = 50) -> dict[str, list[str]]:
    domain_tf: dict[str, Counter] = defaultdict(Counter)
    df: Counter = Counter()
    for chunk in chunks:
        toks = [t for t in tokenize(chunk["text"]) if t not in STOP and len(t) > 2]
        domain_tf[chunk["domain"]].update(toks)
        df.update(set(toks))
    num_domains = max(1, len(domain_tf))
    out: dict[str, list[str]] = {}
    for domain, counts in domain_tf.items():
        scored = []
        for tok, tf in counts.items():
            idf = math.log((1 + num_domains) / (1 + df[tok])) + 1.0
            scored.append((tf * idf, tok))
        out[domain] = [tok for _, tok in sorted(scored, reverse=True)[:top_n]]
    return out


def lexical_gate(query: str, domain_keywords: dict[str, list[str]]) -> dict:
    toks = set(tokenize(query))
    matches = {d: sorted(toks.intersection(words)) for d, words in domain_keywords.items()}
    total = sum(len(v) for v in matches.values())
    support_terms = {"benefit", "benefits", "renew", "renewal", "license", "claim", "registration", "address", "portal", "application", "support"}
    support_like = bool(toks.intersection(support_terms)) or total > 0
    return {"pass": support_like, "matches": matches, "match_count": total}

