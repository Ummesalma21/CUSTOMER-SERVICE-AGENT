# ESA and AQS Evaluation

Source: official Baseline vs Proposed answer-only comparison.
Evaluated answerable rows: `500`
Similarity backend: `sentence_transformer`

Thresholds:

- query_citation_similarity >= `0.35`
- answer_citation_similarity >= `0.4`
- query_answer_similarity >= `0.3`

| Metric | Baseline RAG | Proposed |
|---|---:|---:|
| ESA | 0.4760 | 0.5300 |
| AQS | 0.6270 | 0.6733 |

The proposed system improves ESA from 0.4760 to 0.5300 and AQS from 0.6270 to 0.6733 under the same automatic thresholds.

ESA is a binary automatic proxy for whether the final answer is supported by its cited evidence. AQS is a 0-to-1 automatic rubric averaging fluency, correctness/directness, and trueness/grounding. These are not human-evaluation scores.
