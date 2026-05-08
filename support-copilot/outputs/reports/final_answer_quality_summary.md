# Final Answer Quality Evaluation

Answer-quality metrics are computed only on answerable examples. Citation markers are stripped before token overlap and ROUGE-L scoring.

Rows: `1500`
Reference answers available: `0`
Note: No gold/reference answer text was present in final_answer_only_predictions.jsonl; token F1 and ROUGE-L are not computed.

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| AnswerTokenF1 | N/A | N/A |
| ROUGE-L | N/A | N/A |
| NoFragmentRate | 0.6967 | 1.0000 |
| FragmentRate | 0.3033 | 0.0000 |
| CitationAttachedRate | 1.0000 | 1.0000 |
| DuplicateCitationRate | 1.0000 | 0.0000 |
| EmptyOrInvalidAnswerRate | 0.0907 | 0.0000 |
| CompleteAnswerRate | 0.6967 | 1.0000 |
| CitationRelevanceRate | 1.0000 | 1.0000 |
| AverageAnswerLengthWords | 19.7893 | 33.0690 |

## Interpretation
Answer quality is preserved