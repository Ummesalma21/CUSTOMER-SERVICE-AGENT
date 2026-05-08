# Final Mixed Grounding Metrics

Config: `configs\final_eval_balanced_triage_best.yaml`
Predictions: `outputs/reports/final_mixed_best_predictions.jsonl`
Rows: `1000`

## Workflow Macro-F1
Baseline: `0.24999999999999997`
Proposed: `0.6742703572301131`

## Mixed Workflow Grounding
| Metric | Baseline RAG | Proposed |
|---|---:|---:|
| SupportedResponseRate | 0.5510 | 0.7180 |
| UnsupportedAnswerRate | 0.4490 | 0.2820 |
| EvidenceUseAccuracy | 0.6000 | 0.7650 |

## ANSWER-Only Grounding Kept Separate
Baseline: `{'Recall@5': 0.415, 'MRR@10': 0.3116666666666667, 'EvidenceHit@5': 0.415, 'CitationPrecision': 1.0}`
Proposed: `{'Recall@5': 0.415, 'MRR@10': 0.30869444444444455, 'EvidenceHit@5': 0.415, 'CitationPrecision': 1.0}`

## Claim Supported
Workflow Macro-F1 improves over baseline, and mixed workflow grounding improves because the proposed system avoids more unsupported direct answers on TICKET/REJECT cases while preserving near-baseline answer-only retrieval.