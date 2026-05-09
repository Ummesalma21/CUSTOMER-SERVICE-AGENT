# Fixed Generator Quality

Test file: `data\processed\generator_test.jsonl`
Rows: `960`

| Metric | Extractive fallback | Fixed FLAN-T5 |
|---|---:|---:|
| AnswerTokenF1 | 0.13432655539049873 | 0.16869526587751185 |
| AverageAnswerLengthWords | 15.009375 | 16.801041666666666 |
| CitationAttachedRate | 1.0 | 1.0 |
| EmptyOrInvalidAnswerRate | 0.36770833333333336 | 0.2916666666666667 |
| EvidenceSupportedHeuristicRate | 0.6010416666666667 | 0.6114583333333333 |
| FragmentRate | 0.36770833333333336 | 0.2916666666666667 |
| INSUFFICIENT_EVIDENCE_rate | 0.36770833333333336 | 0.2916666666666667 |
| NoFragmentRate | 0.6322916666666667 | 0.7083333333333334 |
| QueryCopyRate | 0.0 | 0.007291666666666667 |
| QuestionAsAnswerRate | 0.084375 | 0.051041666666666666 |
| ROUGE-L | 0.10483541320514049 | 0.13542655490327943 |