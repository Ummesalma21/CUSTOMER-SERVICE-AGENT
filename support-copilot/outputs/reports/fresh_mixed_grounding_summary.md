# Final Mixed Grounding Metrics

Config: `configs\final_eval_balanced_triage_best.yaml`
Predictions: `outputs\reports\fresh_mixed_best_predictions.jsonl`
Rows: `1000`

## Workflow Macro-F1
Baseline: `0.24999999999999997`
Proposed: `0.5771649777138482`

## Mixed Workflow Grounding
| Metric | Baseline RAG | Proposed |
|---|---:|---:|
| SupportedResponseRate | 0.5750 | 0.6580 |
| UnsupportedAnswerRate | 0.4250 | 0.3190 |
| EvidenceUseAccuracy | 0.6000 | 0.6760 |

## ANSWER-Only Grounding Kept Separate
Baseline: `{'Recall@5': 0.37833333333333335, 'MRR@10': 0.24869444444444436, 'EvidenceHit@5': 0.37833333333333335, 'CitationPrecision': 1.0}`
Proposed: `{'Recall@5': 0.37333333333333335, 'MRR@10': 0.2521388888888888, 'EvidenceHit@5': 0.37333333333333335, 'CitationPrecision': 1.0}`

## Claim Supported
Workflow Macro-F1 improves over baseline, and mixed workflow grounding improves because the proposed system avoids more unsupported direct answers on TICKET/REJECT cases while preserving near-baseline answer-only retrieval.