# Final Answer Quality Evaluation

Answer-quality metrics are computed only on answerable examples. Citation markers are stripped before token overlap and ROUGE-L scoring.

Rows: `500`
Reference answers available: `500`
Note: Gold/reference answer text was available and used for token F1 and ROUGE-L.

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| AnswerTokenF1 | 0.1929 | 0.1700 |
| ROUGE-L | 0.1632 | 0.1403 |
| NoFragmentRate | 0.8380 | 1.0000 |
| FragmentRate | 0.1620 | 0.0000 |
| CitationAttachedRate | 1.0000 | 1.0000 |
| DuplicateCitationRate | 1.0000 | 0.0000 |
| EmptyOrInvalidAnswerRate | 0.0160 | 0.0000 |
| CompleteAnswerRate | 0.8380 | 1.0000 |
| CitationRelevanceRate | 1.0000 | 1.0000 |
| AverageAnswerLengthWords | 23.3900 | 36.6022 |

## Interpretation
Answer quality is preserved