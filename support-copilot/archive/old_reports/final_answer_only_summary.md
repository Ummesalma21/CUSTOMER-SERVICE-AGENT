# Final Answer-Only Evaluation

Config: `configs\final_eval_generator_fixed.yaml`
Eval file: `data/processed/eval_set.jsonl`
Answerable rows: `500`

## Baseline RAG
`{'Recall@1': 0.16, 'Recall@5': 0.36, 'MRR@10': 0.23323333333333332, 'EvidenceHit@5': 0.36, 'CitationPrecision': 1.0, 'CitationRecall': 1.0, 'GroundedAnswerRate': 1.0, 'UnsupportedClaimRate': 0.0}`

## Proposed Balanced Triage
`{'Recall@1': 0.148, 'Recall@5': 0.35, 'MRR@10': 0.2229, 'EvidenceHit@5': 0.35, 'CitationPrecision': 1.0, 'CitationRecall': 1.0, 'GroundedAnswerRate': 1.0, 'UnsupportedClaimRate': 0.0}`