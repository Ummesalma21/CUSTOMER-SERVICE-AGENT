# Fixed Generator Quality

Test file: `data\processed\generator_val.jsonl`
Rows: `960`

| Metric | Extractive fallback | Fixed FLAN-T5 |
|---|---:|---:|
| AnswerTokenF1 | 0.09908536941068943 | 0.1581073974654941 |
| AverageAnswerLengthWords | 16.266666666666666 | 21.680208333333333 |
| CitationAttachedRate | 1.0 | 1.0 |
| EmptyOrInvalidAnswerRate | 0.48541666666666666 | 0.2708333333333333 |
| EvidenceSupportedHeuristicRate | 0.50625 | 0.6260416666666667 |
| FragmentRate | 0.48541666666666666 | 0.2708333333333333 |
| INSUFFICIENT_EVIDENCE_rate | 0.48541666666666666 | 0.2708333333333333 |
| NoFragmentRate | 0.5145833333333333 | 0.7291666666666666 |
| QueryCopyRate | 0.0 | 0.005208333333333333 |
| QuestionAsAnswerRate | 0.05416666666666667 | 0.013541666666666667 |
| ROUGE-L | 0.07496800683515259 | 0.12706365842003037 |