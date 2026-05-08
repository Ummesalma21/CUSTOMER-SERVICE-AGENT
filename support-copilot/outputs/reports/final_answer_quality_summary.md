# Final Answer Quality Evaluation

Answer-quality metrics are computed only on answerable examples. Citation markers are stripped before token overlap and ROUGE-L scoring.

Rows: `1500`
Reference answers available: `0`
Note: No gold/reference answer text was present in final_answer_only_predictions.jsonl; token F1 and ROUGE-L are not computed.

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| AnswerTokenF1 | N/A | N/A |
| ROUGE-L | N/A | N/A |
| NoFragmentRate | 0.6967 | 0.6916 |
| CitationAttachedRate | 1.0000 | 1.0000 |
| DuplicateCitationRate | 1.0000 | 0.9447 |
| EmptyOrInvalidAnswerRate | 0.0907 | 0.0931 |
| AverageAnswerLengthWords | 19.7893 | 16.9683 |

## Interpretation
Answer quality is degraded on fragment-rate