# Fresh Rechunked Full Run Summary

Run date: 2026-05-08.

Archived previous run: `outputs/archive_runs/20260508_225006`.

## Why This Run Was Started

The previous generator was not fine-tuned and some KB chunks began mid-sentence. The fresh run rebuilt processed data with sentence-aware chunks, retrained all major components, created generator training examples from real user-to-agent turns, and trained a small grounded generator checkpoint.

## Fresh Configs

- Data/training: `configs/fresh_rechunked_full.yaml`
- Workflow evaluation: `configs/final_eval_balanced_triage_best.yaml`
- Rechunked calibration trial: `configs/fresh_eval_rechunked_calibrated.yaml`
- Generator inference: `configs/final_eval_generator_finetuned.yaml`

## Archived Artifacts

The following old artifacts were moved under `outputs/archive_runs/20260508_225006`:

- `outputs/retriever`
- `outputs/reranker`
- `outputs/triage`
- `outputs/triage_balanced`
- `outputs/preference`
- `outputs/generator`
- `outputs/reports`
- `outputs/logs`
- `data/processed`
- `data/indexes`
- `checkpoints`

## Dataset And Preprocessing

Preprocessing command:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\fresh_rechunked_full.yaml
```

Fresh data stats:

| Item | Count |
|---|---:|
| KB chunks | 7806 |
| Dialogue turns | 27920 |
| Retriever train pairs | 9000 |
| Reranker train pairs | 36000 |
| Triage train examples | 18000 |
| Preference pairs | 1000 |
| Generator train examples | 10080 |
| Generator validation examples | 960 |
| Generator test examples | 960 |
| Eval set rows | 1000 |

Chunking changed from fixed word windows to sentence-aware chunks with `max_words=90`, `min_words=18`, and `sentence_overlap=1`.

## Checkpoints

| Component | Model | Checkpoint | Notes |
|---|---|---|---|
| Retriever | `sentence-transformers/all-MiniLM-L6-v2` | `outputs/retriever/sentence_transformer` | 9000 pairs, GPU `cuda:0` |
| KB index | SentenceTransformer JSON/FAISS index | `data/indexes/kb_index.json`, `data/indexes/kb_index.faiss` | 7806 chunks |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `outputs/reranker/cross_encoder` | 36000 pairs, 9000 positives, GPU `cuda:0` |
| Initial triage | `distilbert-base-uncased` | `outputs/triage/distilbert` | 18000 examples, no separate validation split in this config |
| Balanced triage | `distilbert-base-uncased` | `outputs/triage_balanced/distilbert` | 18000 train, 2000 validation |
| Preference | Lightweight rubric ranker | `outputs/preference` | 1000 pairs |
| Generator | `google/flan-t5-small` | `outputs/generator/flan_t5` | 10080 train, 960 validation; metrics are unstable, see below |

## Training Results

### Retriever

- Trained pairs: `9000`
- Epochs: `2`
- Train loss: `0.6634`
- Device: `cuda:0`

### Reranker

- Trained pairs: `36000`
- Positives: `9000`
- Epochs: `1`
- Train loss: `0.02967`
- Device: `cuda:0`

### Initial Triage

- Trained examples: `18000`
- Epochs: `4`
- Train-set accuracy: `1.0`
- Train Macro-F1: `1.0`
- Important caveat: this run had no separate validation file, so these are train-set metrics only.

### Balanced Triage

- Train examples: `18000`
- Validation examples: `2000`
- Validation accuracy: `0.999`
- Validation Macro-F1: `0.9990`
- Validation ANSWER F1: `0.9990`
- Validation TICKET F1: `0.9990`
- Validation REJECT F1: `0.9990`
- Validation TBP@0.15: `0.999`

Balanced triage leakage check:

- Mixed-eval exclusion query count: `897`
- Excluded overlapping answer-source rows: `4384`
- Train exact overlap with mixed eval: `0`

### Generator

- Model: `google/flan-t5-small`
- Train examples: `10080`
- Validation examples: `960`
- Epochs: `1`
- Output: `outputs/generator/flan_t5`
- Reported train loss: `0.0`
- Reported eval loss: `NaN`

Generator caveat: the checkpoint was saved, but the training metrics are not trustworthy because the run reported `NaN` gradient norms and `NaN` eval loss. Inference still uses answer-quality validation and falls back to extractive synthesis when generated output is fragmentary or invalid.

## Mixed Workflow Evaluation

Evaluation file: `data/processed/eval_mixed_1000.jsonl`

The mixed ANSWER builder was tightened after the first fresh evaluation because fallback dialogue rows included vague fragments. Final mixed eval contains 600 support-like answerable rows, 200 synthetic ticket rows, and 200 synthetic reject rows.

| Metric | Baseline RAG | Fresh Proposed |
|---|---:|---:|
| Tool Decision Accuracy | 0.600 | 0.676 |
| ANSWER Precision | 0.600 | 0.702 |
| ANSWER Recall | 1.000 | 0.867 |
| ANSWER F1 | 0.750 | 0.776 |
| TICKET Precision | 0.000 | 0.464 |
| TICKET Recall | 0.000 | 0.445 |
| TICKET F1 | 0.000 | 0.454 |
| REJECT Precision | 0.000 | 1.000 |
| REJECT Recall | 0.000 | 0.335 |
| REJECT F1 | 0.000 | 0.502 |
| Macro-F1 | 0.250 | 0.577 |
| FalseRejectRate | 0.000 | 0.000 |
| FalseAcceptRate | 1.000 | 0.665 |
| OODAnswerRate | 1.000 | 0.550 |
| TicketMissRate | 1.000 | 0.555 |

Confusion matrix, proposed:

| Gold \ Pred | ANSWER | TICKET | REJECT |
|---|---:|---:|---:|
| ANSWER | 520 | 80 | 0 |
| TICKET | 111 | 89 | 0 |
| REJECT | 110 | 23 | 67 |

Latency:

| Metric | Value |
|---|---:|
| Average latency | 49.88 ms |
| p95 latency | 62.80 ms |
| Throughput | 20.05 qps |

## Mixed Grounding / Evidence Use

| Metric | Baseline RAG | Fresh Proposed |
|---|---:|---:|
| SupportedResponseRate | 0.575 | 0.658 |
| UnsupportedAnswerRate | 0.425 | 0.319 |
| EvidenceUseAccuracy | 0.600 | 0.676 |

## Answer-Only Retrieval / Grounding

Config: `configs/final_eval_generator_finetuned.yaml`

Rows: `500`

| Metric | Baseline RAG | Fresh Proposed |
|---|---:|---:|
| Recall@1 | 0.160 | 0.156 |
| Recall@5 | 0.360 | 0.362 |
| MRR@10 | 0.233 | 0.232 |
| EvidenceHit@5 | 0.360 | 0.362 |
| CitationPrecision | 1.000 | 1.000 |
| CitationRecall | 1.000 | 1.000 |
| GroundedAnswerRate | 1.000 | 1.000 |
| UnsupportedClaimRate | 0.000 | 0.000 |

Unlike the archived run, answer references are now available for this eval because preprocessing writes `gold_answer` / `reference_answer` from user-to-agent turns or evidence fallback.

## Answer Quality

Reference answers available: `500`.

| Metric | Baseline RAG | Fresh Proposed |
|---|---:|---:|
| AnswerTokenF1 | 0.193 | 0.170 |
| ROUGE-L | 0.163 | 0.140 |
| NoFragmentRate | 0.838 | 1.000 |
| FragmentRate | 0.162 | 0.000 |
| CompleteAnswerRate | 0.838 | 1.000 |
| CitationAttachedRate | 1.000 | 1.000 |
| DuplicateCitationRate | 1.000 | 0.000 |
| EmptyOrInvalidAnswerRate | 0.016 | 0.000 |
| AverageAnswerLengthWords | 23.39 | 36.60 |

Interpretation: answer formatting and fragment control improved, but lexical overlap with reference answers dropped. The fresh generator checkpoint should not be claimed as a quality win; the robust win is cleaner formatting plus validated fallback behavior.

## Demo Checks

Saved to `outputs/reports/fresh_demo_quality_checks.md`.

Observed behavior:

- Benefits renewal: `ANSWER` with one citation from `ssa_renewal_03`.
- IPL: `REJECT` with reason `out_of_domain`.
- Account-specific benefits issue: `TICKET` with `CreateTicket`, category `ssa`.
- Vague query: `REJECT` with reason `underspecified_or_out_of_scope`.

## Honest Limitations

- Sentence-aware chunking improved evidence cleanliness but reduced answer-only retrieval metrics compared with the archived final run.
- The generator fine-tuning run completed but produced unstable metrics (`NaN` eval loss), so it should be treated as an experimental checkpoint, not a reliable quality improvement.
- Mixed workflow still improves over baseline, but less strongly than the archived final run.
- TICKET and REJECT examples remain partly synthetic because MultiDoc2Dial mainly contains answerable support turns.
- The proposed system still prioritizes conservative rejection: REJECT precision is `1.0`, but REJECT recall is only `0.335`.

## Fixed Generator Debug Update

After the unstable generator run above, the generator-only pipeline was debugged without retraining retriever, reranker, triage, the preference ranker, the KB index, or domain centroids.

Fixed checkpoint: `outputs/generator/flan_t5_fixed`

Training config: `configs/generator_fixed_full.yaml`

Final generator demo config: `configs/final_eval_generator_fixed.yaml`

Full generator dataset retained:

| Split | Rows |
|---|---:|
| train | 10080 |
| validation | 960 |
| test | 960 |

Fixed training result:

| Metric | Value |
|---|---:|
| train loss | 3.0252 |
| eval loss | 2.9355 |
| eval perplexity | 18.8308 |
| zero-loss steps | 0 |
| all-ignored labels | 0 |
| device | cuda |

Generator-quality test metrics on `data/processed/generator_test.jsonl`:

| Metric | Extractive fallback | Fixed FLAN-T5 |
|---|---:|---:|
| AnswerTokenF1 | 0.1343 | 0.1687 |
| ROUGE-L | 0.1048 | 0.1354 |
| NoFragmentRate | 0.6323 | 0.7083 |
| FragmentRate | 0.3677 | 0.2917 |
| QuestionAsAnswerRate | 0.0844 | 0.0510 |
| EmptyOrInvalidAnswerRate | 0.3677 | 0.2917 |
| EvidenceSupportedHeuristicRate | 0.6010 | 0.6115 |

Current fixed-generator demo checks are saved to `outputs/reports/generator_fixed_demo_checks.md`. The benefits-renewal demo now returns a complete cited answer from `ssa_renewal_03`, OOD/vague queries reject without random citations, and account-specific renewal issues create a ticket.

Important reporting note: the fixed generator is stable and improves generator-test lexical/formatting metrics over the extractive fallback, but full end-to-end answer-only evaluation with the final guardrail settings timed out after 20 minutes. Therefore, the course-level retrieval/workflow claims should continue to use the completed balanced-triage evaluations unless a longer fixed-generator end-to-end evaluation is run.

## ESA and AQS Evaluation

Output files:

- `outputs/reports/esa_aqs_metrics.json`
- `outputs/reports/esa_aqs_summary.md`
- `outputs/reports/esa_aqs_scored_predictions.jsonl`

Prediction file: `outputs/reports/final_answer_only_generator_fixed_predictions.jsonl`

Note: this older two-way ESA/AQS table used the fine-tuned retriever-only RAG as its baseline. The official assignment baseline is now separated below as Baseline-0 Pretrained RAG.

Evaluated rows: `500` answerable examples.

ESA, Evidence Support Accuracy, is a binary automatic proxy for whether a final answer is actually supported by its cited evidence. A row passes only when a citation exists, cited evidence text is found, the citation is relevant to the query, the answer is supported by that evidence, the answer addresses the query, and the answer is not malformed.

AQS, Answer Quality Score, is a 0-to-1 automatic rubric: `(Fluency + Correctness + Trueness) / 6`, where each component is scored 0, 1, or 2.

The same sentence-transformer similarity thresholds were used for baseline and proposed:

- query-citation similarity >= `0.35`
- answer-citation similarity >= `0.40`
- query-answer similarity >= `0.30`

| Metric | Baseline RAG | Proposed |
|---|---:|---:|
| ESA | 0.7480 | 0.5300 |
| AQS | 0.7840 | 0.6733 |

Interpretation: proposed did not improve ESA or AQS on this answer-only prediction file. This is an important limitation. The proposed system's stronger results remain in workflow control and mixed evidence-use behavior, while answer synthesis/evidence support for answerable-only cases still needs better generator training and calibration. These ESA/AQS numbers are automatic proxy metrics, not human evaluation.

## Baselines

Output files:

- `outputs/reports/baseline_pretrained_metrics.json`
- `outputs/reports/baseline_finetuned_metrics.json`
- `outputs/reports/three_way_final_comparison.json`
- `outputs/reports/three_way_final_comparison.md`

The official assignment baseline is now **Baseline-0: Pretrained RAG**. It uses `sentence-transformers/all-MiniLM-L6-v2` directly, searches the full KB, does not use domain routing, reranking, triage/tool-policy, CreateTicket, RejectQuery, or the preference/rubric ranker, and always returns an ANSWER with cited retrieved evidence.

**Baseline-1: Fine-tuned RAG** is kept only as an ablation. It uses the fine-tuned retriever/index but still has no tools, no triage, no reject behavior, no ticket behavior, no reranker, and no generator.

**Proposed** adds domain routing, triage/tool-policy, ticketing, rejection, and grounded answer validation/generation.

Answer-only evaluation, 500 answerable examples:

| System | Recall@1 | Recall@5 | MRR@10 | EvidenceHit@5 | CitationPrecision | GroundedAnswerRate | UnsupportedClaimRate | ESA | AQS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-0 Pretrained RAG | 0.1040 | 0.1820 | 0.1291 | 0.1820 | 1.0000 | 1.0000 | 0.0000 | 0.4760 | 0.6270 |
| Baseline-1 Fine-tuned RAG | 0.1580 | 0.3620 | 0.2327 | 0.3620 | 1.0000 | 1.0000 | 0.0000 | 0.7480 | 0.7840 |
| Proposed | 0.1560 | 0.3620 | 0.2320 | 0.3620 | 1.0000 | 1.0000 | 0.0000 | 0.5300 | 0.6733 |

Mixed workflow evaluation, 1000 examples:

| System | ToolAcc | ANSWER F1 | TICKET F1 | REJECT F1 | Macro-F1 | SupportedResponseRate | UnsupportedAnswerRate | EvidenceUseAccuracy | OODAnswerRate | TicketMissRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-0 Pretrained RAG | 0.6000 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5750 | 0.4250 | 0.6000 | 1.0000 | 1.0000 |
| Baseline-1 Fine-tuned RAG | 0.6000 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5750 | 0.4250 | 0.6000 | 1.0000 | 1.0000 |
| Proposed | 0.6760 | 0.7755 | 0.4541 | 0.5019 | 0.5772 | 0.6580 | 0.3190 | 0.6760 | 0.5500 | 0.5550 |

Honest interpretation: proposed clearly improves mixed workflow metrics over both simple RAG baselines. On answer-only retrieval, proposed improves substantially over Baseline-0 and is essentially tied with Baseline-1 on Recall@5/EvidenceHit@5, but it does not beat Baseline-1 on ESA or AQS. The fine-tuned RAG ablation remains stronger on those automatic answer-only support/quality proxies.

## Preference/Rubric Alignment Scores

Output files:

- `outputs/reports/preference_score_comparison.json`
- `outputs/reports/preference_score_comparison.md`

Preference/rubric alignment exists as a lightweight pairwise rubric ranker, not DPO or RLHF.

Training artifact:

- `data/processed/preference_pairs.jsonl`: 1000 pairs
- `outputs/preference/metrics.json`: `{"trained_pairs": 1000, "pair_accuracy": 1.0, "model": "rubric-ranker"}`
- `outputs/preference/model.json`: `{"rubric": "citation+grounding+tool+concise"}`

The scorer is implemented in `src/preference/score_candidates.py`. It rewards inline or attached citation markers, insufficient-evidence ticket-style language, out-of-domain rejection-style language, concise answers, and an extra citation bonus for ANSWER examples. For this comparison, structured proposed citations were converted to the same citation-marker form before scoring, so formatting differences do not unfairly penalize the proposed system.

| Evaluation | System | Mean Preference Score | Win Rate vs Baseline-0 |
|---|---|---:|---:|
| Answer-only | Baseline-0 Pretrained RAG | 4.9680 | - |
| Answer-only | Baseline-1 Fine-tuned RAG | 4.9760 | 0.0240 |
| Answer-only | Proposed | 4.1200 | 0.0200 |
| Mixed workflow | Baseline-0 Pretrained RAG | 4.5810 | - |
| Mixed workflow | Baseline-1 Fine-tuned RAG | 4.5690 | 0.0140 |
| Mixed workflow | Proposed | 3.9990 | 0.0160 |

Honest interpretation: the rubric alignment step is present and trained/evaluated, but this simple scalar rubric is biased toward cited direct ANSWER strings. It does not fully capture the value of TICKET/REJECT decisions in mixed workflow settings, where the proposed system improves Macro-F1 and unsupported-answer avoidance. Therefore, preference score should be reported as an alignment component/ablation, not as the main evidence of proposed-system improvement.

## Official Baseline-0 vs Proposed Final Comparison

Output files:

- `outputs/reports/baseline0_vs_proposed_metrics.json`
- `outputs/reports/baseline0_vs_proposed_summary.md`
- `outputs/reports/baseline0_vs_proposed_predictions.jsonl`
- `outputs/reports/unsupported_answer_safety_metrics.json`
- `outputs/reports/unsupported_answer_safety_summary.md`
- `outputs/reports/baseline0_vs_proposed_latency.json`
- `outputs/reports/baseline0_vs_proposed_efficiency.json`

Baseline-0 is the official simple pretrained RAG baseline. It uses `sentence-transformers/all-MiniLM-L6-v2`, full KB search, no reranker, no routing, no triage/tool-policy, no ticketing, no rejection, no preference/rubric ranker, and always returns `ANSWER` with a citation.

Proposed is the final routed/tool-using support copilot with trained/fine-tuned components, domain routing, triage/tool-policy, ticketing, rejection, and grounded answer validation/generation.

Answer-only retrieval and grounding:

| Metric | Baseline-0 Pretrained RAG | Proposed | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.1040 | 0.1560 | +0.0520 |
| Recall@5 | 0.1820 | 0.3620 | +0.1800 |
| MRR@10 | 0.1291 | 0.2320 | +0.1028 |
| EvidenceHit@5 | 0.1820 | 0.3620 | +0.1800 |
| CitationPrecision | 1.0000 | 1.0000 | +0.0000 |
| GroundedAnswerRate | 1.0000 | 1.0000 | +0.0000 |
| UnsupportedClaimRate | 0.0000 | 0.0000 | +0.0000 |

ESA/AQS:

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| ESA | 0.4760 | 0.5300 | +0.0540 |
| AQS | 0.6270 | 0.6733 | +0.0463 |

Unsupported-answer safety:

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| UnsupportedAnswerRate | 1.0000 | 0.5525 | -0.4475 |
| UnsupportedAnswerCount | 400 | 221 | -179 |
| UnsupportedAnswerPreventionCount | - | 179 | - |
| UnsupportedAnswerPreventionRate | - | 0.4475 | - |
| SafeActionRate | 0.0000 | 0.4475 | +0.4475 |
| OODAnswerRate | 1.0000 | 0.5500 | -0.4500 |
| TicketMissRate | 1.0000 | 0.5550 | -0.4450 |
| FalseRejectOnAnswerableRate | 0.0000 | 0.0000 | +0.0000 |

Mixed workflow:

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| Tool Decision Accuracy | 0.6000 | 0.6760 | +0.0760 |
| ANSWER F1 | 0.7500 | 0.7755 | +0.0255 |
| TICKET F1 | 0.0000 | 0.4541 | +0.4541 |
| REJECT F1 | 0.0000 | 0.5019 | +0.5019 |
| Macro-F1 | 0.2500 | 0.5772 | +0.3272 |
| SupportedResponseRate | 0.5750 | 0.6580 | +0.0830 |
| UnsupportedAnswerRate | 0.4250 | 0.3190 | -0.1060 |
| EvidenceUseAccuracy | 0.6000 | 0.6760 | +0.0760 |

Efficiency and tool usage:

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| avg_latency_ms | N/A | 42.2952 | - |
| p95_latency_ms | N/A | 56.8940 | - |
| throughput_qps | N/A | 23.6433 | - |
| avg_fraction_kb_searched | 1.0000 | 0.9172 | -0.0828 |
| global_fallback_rate | 1.0000 | 0.8720 | -0.1280 |
| avg_num_domains_searched | 0.0000 | 2.7990 | +2.7990 |
| avg_num_tool_calls | 1.0000 | 5.7730 | +4.7730 |
| REE@5 | 0.1820 | 0.3947 | +0.2127 |

Latency note: Baseline-0 live per-query latency was not remeasured in this final comparison because its predictions were batch-generated in `three_way_final_comparison`. Proposed latency is loaded from saved per-query mixed-eval traces. Do not claim proposed is faster than Baseline-0.

## Unsupported Answer Prevention

Unsupported case = gold `TICKET` or gold `REJECT`.

Unsupported answer = system returns direct `ANSWER` on an unsupported case.

UnsupportedAnswerPreventionRate = fraction of Baseline-0 unsupported answers where Proposed instead chose `TICKET` or `REJECT`.

Since Baseline-0 is a simple RAG system without ticket or reject tools, direct triage-F1 comparison can be misleading. The fairer customer-support safety comparison is unsupported-answer safety: how often a system gives a direct answer when the KB does not contain sufficient evidence, and how many such failures the proposed system prevents.

Baseline-0 answered all 400 unsupported cases, so its unsupported-case `UnsupportedAnswerRate` is `1.0000`. Proposed answered 221 unsupported cases directly and prevented 179 of Baseline-0's unsupported answers, for an `UnsupportedAnswerPreventionRate` of `0.4475`.

Baseline-1 Fine-tuned RAG remains an ablation and is not the official assignment baseline. It remains strong on answer-only extraction, but it still lacks ticket/reject tools.
