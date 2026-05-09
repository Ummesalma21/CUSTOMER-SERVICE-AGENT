# Mixed Evaluation Summary

Config: `configs\fresh_eval_rechunked_calibrated.yaml`
Eval file: `data/processed/eval_mixed_1000.jsonl`

## Counts
`{'total': 1000, 'TICKET': 200, 'synthetic_ticket': 200, 'ANSWER': 600, 'multidoc2dial': 600, 'REJECT': 200, 'synthetic_reject': 200}`

## Baseline RAG
`{'Tool Decision Accuracy': 0.6, 'ANSWER F1': 0.7499999999999999, 'ANSWER Precision': 0.6, 'ANSWER Recall': 1.0, 'TICKET F1': 0.0, 'TICKET Precision': 0.0, 'TICKET Recall': 0.0, 'REJECT F1': 0.0, 'REJECT Precision': 0.0, 'REJECT Recall': 0.0, 'confusion_matrix': {'ANSWER': {'ANSWER': 600, 'TICKET': 0, 'REJECT': 0}, 'TICKET': {'ANSWER': 200, 'TICKET': 0, 'REJECT': 0}, 'REJECT': {'ANSWER': 200, 'TICKET': 0, 'REJECT': 0}}, 'Macro-F1': 0.24999999999999997, 'FalseRejectRate': 0.0, 'FalseAcceptRate': 1.0, 'OODAnswerRate': 1.0, 'TicketMissRate': 1.0}`

## Proposed Calibrated System
`{'Tool Decision Accuracy': 0.675, 'ANSWER F1': 0.7749627421758569, 'ANSWER Precision': 0.7008086253369272, 'ANSWER Recall': 0.8666666666666667, 'TICKET F1': 0.4540816326530613, 'TICKET Precision': 0.4635416666666667, 'TICKET Recall': 0.445, 'REJECT F1': 0.49624060150375937, 'REJECT Precision': 1.0, 'REJECT Recall': 0.33, 'confusion_matrix': {'ANSWER': {'ANSWER': 520, 'TICKET': 80, 'REJECT': 0}, 'TICKET': {'ANSWER': 111, 'TICKET': 89, 'REJECT': 0}, 'REJECT': {'ANSWER': 111, 'TICKET': 23, 'REJECT': 66}}, 'Macro-F1': 0.5750949921108925, 'FalseRejectRate': 0.0, 'FalseAcceptRate': 0.67, 'OODAnswerRate': 0.555, 'TicketMissRate': 0.555}`

## ANSWER-Only Retrieval
Baseline: `{'Recall@5': 0.37833333333333335, 'MRR@10': 0.24869444444444436, 'EvidenceHit@5': 0.37833333333333335, 'CitationPrecision': 1.0}`
Proposed: `{'Recall@5': 0.37, 'MRR@10': 0.2500555555555555, 'EvidenceHit@5': 0.37, 'CitationPrecision': 1.0}`

## Latency
`{'avg_ms': 49.62365839979611, 'p95_ms': 61.38169998303056, 'qps': 20.151678297142812}`

## Notes
Baseline RAG is answer-only, so TICKET and REJECT examples are counted as decision errors.
This mixed evaluation does not retrain any model; it reuses the configured checkpoints and calibrated inference logic.