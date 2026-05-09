# Generator Debug Final Summary

## Scope

Only the grounded generator pipeline was changed in this pass. Retriever, reranker, triage, balanced triage, preference/rubric ranker, KB index, and domain centroids were not retrained.

## Original Problem

The previous FLAN-T5-small generator run was not reliable:

- train loss was reported as `0.0`
- eval loss became `NaN`
- gradient norms became `NaN`

That pointed to a training-loop, fp16, label construction, or data-format issue rather than a reason to immediately shrink the dataset.

## Dataset Audit

The full existing generator dataset was inspected and kept:

| Split | Rows | Avg target tokens | Target <6 words | Broken spacing |
|---|---:|---:|---:|---:|
| train | 10080 | 29.36 | 23 | 48 |
| val | 960 | 29.55 | 2 | 0 |
| test | 960 | 33.03 | 1 | 0 |

The data quality report is saved at `outputs/reports/generator_data_quality_report.md`. The dataset is imperfect but usable; it was not reduced by default.

## Tokenization And Label Diagnostics

Batch diagnostics confirmed that T5 labels are valid:

- Train rows checked: `24`
- Validation rows checked: `24`
- All-ignored label examples: `0`
- Invalid input ID examples: `0`
- Padding is converted to `-100`, while non-padding target tokens remain trainable.

Detailed diagnostics are saved at `outputs/reports/generator_batch_debug.md` and `outputs/reports/generator_batch_debug_examples.jsonl`.

## Training Fix

`src/generation/train_generator_lora.py` was replaced with a stable explicit seq2seq training loop while preserving the public training entry point.

Important fixes:

- fp16 disabled for the stable run
- explicit target tokenization and pad-token-to-`-100` label masking
- all-ignored-label checks before training
- gradient clipping
- NaN/Inf loss and gradient detection
- zero-loss-step tracking
- diagnostics written when failures occur

## Overfit Sanity Check

The tiny overfit check used 96 examples only as a debugging test, not as the final dataset.

| Metric | Value |
|---|---:|
| train examples | 96 |
| epochs | 5 |
| train loss | 2.9446 |
| eval loss | 2.5046 |
| eval perplexity | 12.2391 |
| zero-loss steps | 0 |
| all-ignored labels | 0 |

Output files:

- `outputs/reports/generator_overfit_check.md`
- `outputs/reports/generator_overfit_predictions.jsonl`
- `outputs/reports/generator_overfit_train_metrics.json`

## Full Fixed Generator Training

The fixed generator was trained on the full existing dataset.

| Setting | Value |
|---|---|
| model | `google/flan-t5-small` |
| checkpoint | `outputs/generator/flan_t5_fixed` |
| train examples | 10080 |
| validation examples | 960 |
| epochs | 1 |
| batch size | 2 |
| gradient accumulation | 8 |
| learning rate | 5e-5 |
| fp16 | false |
| device | cuda |

| Metric | Value |
|---|---:|
| train loss | 3.0252 |
| eval loss | 2.9355 |
| eval perplexity | 18.8308 |
| zero-loss steps | 0 |
| optimizer steps | 630 |
| all-ignored labels | 0 |
| runtime seconds | 2507.81 |

Training metrics are saved at `outputs/reports/generator_fixed_full_train_metrics.json`.

## Generator Quality Metrics

Evaluation used the full `data/processed/generator_test.jsonl` split with 960 examples.

| Metric | Extractive fallback | Fixed FLAN-T5 |
|---|---:|---:|
| AnswerTokenF1 | 0.1343 | 0.1687 |
| ROUGE-L | 0.1048 | 0.1354 |
| NoFragmentRate | 0.6323 | 0.7083 |
| FragmentRate | 0.3677 | 0.2917 |
| QuestionAsAnswerRate | 0.0844 | 0.0510 |
| QueryCopyRate | 0.0000 | 0.0073 |
| EmptyOrInvalidAnswerRate | 0.3677 | 0.2917 |
| AverageAnswerLengthWords | 15.0094 | 16.8010 |
| EvidenceSupportedHeuristicRate | 0.6010 | 0.6115 |
| INSUFFICIENT_EVIDENCE rate | 0.3677 | 0.2917 |
| CitationAttachedRate | 1.0000 | 1.0000 |

The fixed generator improves lexical answer overlap and reduces fragments versus the extractive fallback on the generator test set. These are still heuristic/reference-overlap metrics; they do not prove semantic correctness.

Output files:

- `outputs/reports/generator_fixed_quality_metrics.json`
- `outputs/reports/generator_fixed_quality_summary.md`
- `outputs/reports/generator_fixed_quality_predictions.jsonl`

## End-To-End Answer-Only Check

An end-to-end answer-only rerun with `configs/final_eval_generator_fixed.yaml` was attempted after the final guardrail changes, but it timed out after 20 minutes. The previously completed generator-fixed answer-only files remain in `outputs/reports/`, but they should not be treated as refreshed after the final `top_evidence_count: 1` and stricter validation change.

This means the fixed generator is validated at the generator-test level and by demo checks, but the final course-level retrieval/grounding claims should still rely on the already completed balanced-triage evaluation unless a longer answer-only generator run is completed.

## Inference Changes

`configs/final_eval_generator_fixed.yaml` now uses:

- `generation.model_name: outputs/generator/flan_t5_fixed`
- `generation.top_evidence_count: 1`
- extractive fallback if the generator is unavailable or invalid

The generator is used only for ANSWER synthesis. It does not choose ANSWER, TICKET, or REJECT. Citations are still attached by the system from retrieved evidence.

Postprocessing blocks:

- fragment answers
- long run-on generations without sentence-ending punctuation
- repeated sentence outputs
- question-as-answer outputs
- query-copy outputs
- malformed spacing
- generations with weak query-term coverage
- generations with weak evidence-token support

## Demo Checks

The current demo checks are saved at `outputs/reports/generator_fixed_demo_checks.md`.

Observed behavior:

- `Can I renew my benefits online?` -> ANSWER with a complete benefits-renewal answer and one citation.
- `Who won the IPL yesterday?` -> REJECT with `RejectQuery`.
- Account-specific pending benefits query -> TICKET with `CreateTicket`.
- `Why am I here?` -> REJECT/clarification, no random KB citation.
- `are you trying to lighten the mood?` -> REJECT/clarification, no random KB citation.
- Scholarship query -> ANSWER with student-aid citation.

## Recommendation

The fixed FLAN-T5-small generator is now stable enough to use for demo and as an experimental ANSWER synthesis option:

- finite train and eval losses
- no zero-loss training
- no all-ignored labels
- full dataset retained
- better generator-test answer-quality metrics than extractive fallback
- cleaner demo answers after evidence-selection and validation guardrails

For final report claims, keep the wording conservative:

> Fixed FLAN-T5-small is used only for grounded ANSWER synthesis when enabled; tool decisions remain from the triage/tool-policy model, and citations are attached by the retrieval system. Generator quality improved over the extractive fallback on reference-overlap and formatting metrics, but answer semantic quality remains a limitation and should not be overclaimed.

## Remaining Risks

- The generator still depends on retrieved evidence quality.
- The fixed generator is slower than extractive fallback; demo latencies are around 9-12 seconds per query when models are loaded per CLI process.
- Full end-to-end answer-only evaluation with the fixed generator timed out at 20 minutes after the final guardrail update.
- Some references in the generator dataset are noisy because they originate from dialogue/evidence extraction rather than human-written standalone answers.
