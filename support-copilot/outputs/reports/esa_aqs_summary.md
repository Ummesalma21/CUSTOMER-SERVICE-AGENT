# ESA and AQS Evaluation

Prediction file: `C:\Users\ummes\OneDrive\Documents\New project\CUSTOMER-SERVICE-AGENT\support-copilot\outputs\reports\final_answer_only_generator_fixed_predictions.jsonl`
Evaluated answerable rows: `500`
Similarity backend: `sentence_transformer`

Thresholds:

- query_citation_similarity >= `0.35`
- answer_citation_similarity >= `0.4`
- query_answer_similarity >= `0.3`

| Metric | Baseline RAG | Proposed |
|---|---:|---:|
| ESA | 0.7480 | 0.5300 |
| AQS | 0.7840 | 0.6733 |

Proposed ESA/AQS did not improve on both metrics; ESA is 0.7480 vs 0.5300, and AQS is 0.7840 vs 0.6733. This remains a limitation.

ESA is a binary automatic proxy for whether the final answer is supported by its cited evidence. AQS is a 0-to-1 automatic rubric averaging fluency, correctness/directness, and trueness/grounding. These are not human-evaluation scores.