# Final Comparison: Baseline vs Proposed Domain-Router

## Selected Proposed Thresholds

- top_k_domains: `2`
- min_domain_confidence: `0.7`
- min_candidate_similarity: `0.4`
- min_domain_candidates: `5`
- fallback_merge_mode: `merge`
- rerank_after_merge: `True`

## Final Metrics

| Metric | Baseline | Proposed (Domain-Router) |
|---|---:|---:|
| Recall@5 | 0.1820 | 0.3473 |
| EvidenceHit@5 | 0.1820 | 0.3473 |
| ESA | 0.4760 | 0.7936 |
| AQS | 0.6270 | 0.8195 |
| Macro-F1 | 0.2500 | 0.8257 |
| UnsupportedAnswerRate | 1.0000 | 0.2250 |
| OODAnswerRate | 1.0000 | 0.4500 |
| TicketMissRate | 1.0000 | 0.0000 |