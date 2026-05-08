# Final Answer-Only Evaluation

Config: `configs\final_eval_generator.yaml`
Eval file: `data/processed/eval_set.jsonl`
Answerable rows: `1500`

## Baseline RAG
`{'Recall@1': 0.25666666666666665, 'Recall@5': 0.452, 'MRR@10': 0.32956666666666606, 'EvidenceHit@5': 0.452, 'CitationPrecision': 1.0, 'CitationRecall': 1.0, 'GroundedAnswerRate': 1.0, 'UnsupportedClaimRate': 0.0}`

## Proposed Balanced Triage
`{'Recall@1': 0.254, 'Recall@5': 0.4513333333333333, 'MRR@10': 0.32745555555555494, 'EvidenceHit@5': 0.4513333333333333, 'CitationPrecision': 1.0, 'CitationRecall': 1.0, 'GroundedAnswerRate': 1.0, 'UnsupportedClaimRate': 0.0}`