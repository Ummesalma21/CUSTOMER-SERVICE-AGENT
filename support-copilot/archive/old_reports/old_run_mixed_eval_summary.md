# Mixed Evaluation Summary

Config: `configs/final_eval_mixed_best.yaml`
Eval file: `data/processed/eval_mixed_1000.jsonl`

## Counts
`{'total': 1000, 'TICKET': 200, 'synthetic_ticket': 200, 'ANSWER': 600, 'multidoc2dial': 600, 'REJECT': 200, 'synthetic_reject': 200}`

## Baseline RAG
`{'Tool Decision Accuracy': 0.6, 'ANSWER F1': 0.7499999999999999, 'TICKET F1': 0.0, 'REJECT F1': 0.0, 'Macro-F1': 0.24999999999999997, 'FalseRejectRate': 0.0, 'FalseAcceptRate': 1.0, 'OODAnswerRate': 1.0, 'TicketMissRate': 1.0}`

## Proposed Calibrated System
`{'Tool Decision Accuracy': 0.693, 'ANSWER F1': 0.7729970326409494, 'TICKET F1': 0.6137931034482759, 'REJECT F1': 0.4585635359116022, 'Macro-F1': 0.6151178906669424, 'FalseRejectRate': 0.09875, 'FalseAcceptRate': 0.585, 'OODAnswerRate': 0.58, 'TicketMissRate': 0.555}`

## ANSWER-Only Retrieval
Baseline: `{'Recall@5': 0.415, 'MRR@10': 0.3116666666666667, 'EvidenceHit@5': 0.415, 'CitationPrecision': 1.0}`
Proposed: `{'Recall@5': 0.4, 'MRR@10': 0.3004444444444445, 'EvidenceHit@5': 0.4, 'CitationPrecision': 1.0}`

## Notes
Baseline RAG is answer-only, so TICKET and REJECT examples are counted as decision errors.
This mixed evaluation does not retrain any model; it reuses the configured checkpoints and calibrated inference logic.