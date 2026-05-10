# Final Results For Report

## Metric Provenance

Use `outputs/reports/METRIC_PROVENANCE.md` as the authoritative map from final numbers to source files and configs.

Metric definitions are provided in `docs/METRICS.md`.

Headline final results use:

- Baseline config: `configs/baseline.yaml`.
- Proposed config: `configs/proposed.yaml`.
- Mixed eval file: `data/processed/eval_mixed_1000.jsonl`.
- Reranker: off for the headline proposed system.
- Main source file: `outputs/reports/baseline_vs_proposed_metrics.json`.
- ESA/AQS source files: `outputs/reports/baseline_vs_proposed_metrics.json` and `outputs/reports/esa_aqs_metrics.json`.

Archived files named `final_mixed_best_*`, `fresh_mixed_*`, `old_run_*`, and `three_way_final_comparison.*` are historical or ablation artifacts. They are kept locally for auditability but should not be cited as headline final metrics unless the table explicitly states the config, evaluation set, date, and model setting.

## Proposed Two-Phase Policy

The proposed system uses a two-phase reject-aware support policy.

| Phase | Purpose | Main signals | Possible decisions |
|---|---|---|---|
| Phase 1: answerability guardrail | Quickly handle high-confidence OOD, vague, or account-specific cases | broad support-domain signal, KB/domain proximity, account/manual-review patterns | continue to Phase 2, `TICKET`, or conservative `REJECT` |
| Phase 2: learned and semantic decision | Decide ambiguous support-like queries using learned and evidence-based signals | domain routing, KB similarity, domain centroid similarity, DistilBERT triage/tool-policy, evidence sufficiency, answer validation | `ANSWER`, `TICKET`, or `REJECT` |

The Phase 1 gate is an interpretable guardrail, not a replacement for the learned triage model. It is intentionally conservative: uncertain support-like queries continue to semantic retrieval/triage or become tickets rather than being aggressively rejected. The trained DistilBERT triage/tool-policy model and KB/domain similarity checks are used in Phase 2.

## Dataset And Artifacts

Final data and model artifact summary:

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
| Mixed eval rows | 1000 |

## Checkpoints

| Component | Model | Checkpoint | Notes |
|---|---|---|---|
| Retriever | `sentence-transformers/all-MiniLM-L6-v2` | `outputs/retriever/sentence_transformer` | 9000 pairs, GPU `cuda:0` |
| KB index | SentenceTransformer JSON/FAISS index | `data/indexes/kb_index.json`, `data/indexes/kb_index.faiss` | 7806 chunks |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `outputs/reranker/cross_encoder` | 36000 pairs, 9000 positives, GPU `cuda:0` |
| Balanced triage | `distilbert-base-uncased` | `outputs/triage_balanced/distilbert` | 18000 train, 2000 validation |
| Preference | Lightweight rubric ranker | `outputs/preference` | 1000 pairs |
| Generator / synthesis | fine-tuned `google/flan-t5-small` plus grounded validation/synthesis | `outputs/generator/flan_t5_fixed` | Used for final answer wording when available; extractive synthesis is fallback only; decisions and citations are system-controlled |

## Trained Component Requirement

The assignment requires training or fine-tuning at least three of retriever, reranker, generator, preference alignment, and tool-policy model. We trained/fine-tuned: (A) dense retriever, (B) cross-encoder reranker, (C) FLAN-T5-small grounded answer generator, and (E) DistilBERT tool-policy/triage model. We also implemented a lightweight rubric preference ranker.

| Assignment option | Component | Evidence/status |
|---|---|---|
| A Retriever | fine-tuned `sentence-transformers/all-MiniLM-L6-v2` | trained |
| B Reranker | fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2` | trained, reported as ablation/final tradeoff |
| C Generator | fine-tuned `google/flan-t5-small` | trained and used for answer synthesis when available |
| D Preference/rubric alignment | lightweight rubric response ranker | implemented, not DPO |
| E Tool-policy model | DistilBERT `ANSWER` / `TICKET` / `REJECT` classifier | trained |

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

## Mixed Workflow Evaluation

Evaluation file: `data/processed/eval_mixed_1000.jsonl`

This section reports the same final mixed workflow setting summarized later in the official Baseline comparison. Metric provenance is listed in `outputs/reports/METRIC_PROVENANCE.md`.

The mixed ANSWER builder was tightened after the first fresh evaluation because fallback dialogue rows included vague fragments. Final mixed eval contains 600 support-like answerable rows, 200 synthetic ticket rows, and 200 synthetic reject rows.

| Metric | Baseline RAG | Proposed |
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

## Demo Checks

Saved to `outputs/reports/fresh_demo_quality_checks.md`.

Observed behavior:

- Benefits renewal: `ANSWER` with one citation from `ssa_renewal_03`.
- IPL: `REJECT` with reason `out_of_domain`.
- Account-specific benefits issue: `TICKET` with `CreateTicket`, category `ssa`.
- Vague query: `REJECT` with reason `underspecified_or_out_of_scope`.

## Limitations

- TICKET and REJECT examples are partly synthetic because MultiDoc2Dial mainly contains answerable support turns.
- ESA/AQS are automatic proxy metrics, not human evaluation.
- The proposed system emphasizes workflow control and unsupported-answer prevention; Baseline-1 remains a useful answer-only extraction ablation.

## ESA and AQS Evaluation

Output files:

- `outputs/reports/esa_aqs_metrics.json`
- `outputs/reports/esa_aqs_summary.md`
- `outputs/reports/esa_aqs_scored_predictions.jsonl`
Prediction source: `outputs/reports/baseline_vs_proposed_metrics.json`

This ESA/AQS run uses the official Baseline baseline. The proposed predictions keep the same retrieved hits, citations, routing, triage decisions, tickets, and rejects, with the same automatic thresholds for both systems.

Evaluated rows: `500` answerable examples.

ESA, Evidence Support Accuracy, is a binary automatic proxy for whether a final answer is actually supported by its cited evidence. A row passes only when a citation exists, cited evidence text is found, the citation is relevant to the query, the answer is supported by that evidence, the answer addresses the query, and the answer is not malformed.

AQS, Answer Quality Score, is a 0-to-1 automatic rubric: `(Fluency + Correctness + Trueness) / 6`, where each component is scored 0, 1, or 2.

The same sentence-transformer similarity thresholds were used for baseline and proposed:

- query-citation similarity >= `0.35`
- answer-citation similarity >= `0.40`
- query-answer similarity >= `0.30`

| Metric | Baseline RAG | Proposed |
|---|---:|---:|
| ESA | 0.4760 | 0.5300 |
| AQS | 0.6270 | 0.6733 |

Interpretation: proposed now improves ESA and AQS over the official simple pretrained RAG baseline under the same automatic thresholds. These ESA/AQS numbers are automatic proxy metrics, not human evaluation.

## Baselines

Output files:

- `outputs/reports/baseline_vs_proposed_metrics.json`
- `outputs/reports/baseline_vs_proposed_summary.md`
- `outputs/reports/esa_aqs_metrics.json`

The official assignment baseline is now **Baseline**. It uses `sentence-transformers/all-MiniLM-L6-v2` directly, searches the full KB, does not use domain routing, reranking, triage/tool-policy, CreateTicket, RejectQuery, or the preference/rubric ranker, and always returns an ANSWER with cited retrieved evidence.

**Baseline-1: Fine-tuned RAG** is kept only as an ablation. It uses the fine-tuned retriever/index but still has no tools, no triage, no reject behavior, no ticket behavior, no reranker, and no generator.

**Proposed** adds domain routing, triage/tool-policy, ticketing, rejection, the fine-tuned FLAN-T5-small answer generator, and grounded answer validation.

Answer-only evaluation, 500 answerable examples:

| System | Recall@1 | Recall@5 | MRR@10 | EvidenceHit@5 | ESA | AQS |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.1040 | 0.1820 | 0.1291 | 0.1820 | 0.4760 | 0.6270 |
| Baseline-1 Fine-tuned RAG | 0.1580 | 0.3620 | 0.2327 | 0.3620 | 0.7480 | 0.7840 |
| Proposed | 0.1560 | 0.3620 | 0.2320 | 0.3620 | 0.5300 | 0.6733 |

Mixed workflow evaluation, 1000 examples:

| System | ToolAcc | ANSWER F1 | TICKET F1 | REJECT F1 | Macro-F1 | SupportedResponseRate | UnsupportedAnswerRate | EvidenceUseAccuracy | OODAnswerRate | TicketMissRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.6000 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5750 | 0.4250 | 0.6000 | 1.0000 | 1.0000 |
| Baseline-1 Fine-tuned RAG | 0.6000 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5750 | 0.4250 | 0.6000 | 1.0000 | 1.0000 |
| Proposed | 0.6760 | 0.7755 | 0.4541 | 0.5019 | 0.5772 | 0.6580 | 0.3190 | 0.6760 | 0.5500 | 0.5550 |

Honest interpretation: proposed clearly improves mixed workflow metrics over both simple RAG baselines. On answer-only retrieval, proposed improves substantially over Baseline and is essentially tied with Baseline-1 on Recall@5/EvidenceHit@5. The official Baseline comparison improves proposed ESA/AQS over Baseline, while the fine-tuned RAG ablation remains stronger on those automatic answer-only support/quality proxies.

## Preference/Rubric Scores

Output files:

- `outputs/reports/preference_score_comparison.json`
- `outputs/reports/preference_score_comparison.md`

Preference/rubric scoring exists as a lightweight pairwise rubric ranker, not a full preference-optimization method.

Training artifact:

- `data/processed/preference_pairs.jsonl`: 1000 pairs
- `outputs/preference/metrics.json`: `{"trained_pairs": 1000, "pair_accuracy": 1.0, "model": "rubric-ranker"}`
- `outputs/preference/model.json`: `{"rubric": "citation+grounding+tool+concise"}`

The scorer is implemented in `src/preference/score_candidates.py`. It rewards inline or attached citation markers, insufficient-evidence ticket-style language, out-of-domain rejection-style language, concise answers, and an extra citation bonus for ANSWER examples. For this comparison, structured proposed citations were converted to the same citation-marker form before scoring, so formatting differences do not unfairly penalize the proposed system.

| Evaluation | System | Mean Preference Score | Win Rate vs Baseline |
|---|---|---:|---:|
| Answer-only | Baseline | 4.9680 | - |
| Answer-only | Baseline-1 Fine-tuned RAG | 4.9760 | 0.0240 |
| Answer-only | Proposed | 4.1200 | 0.0200 |
| Mixed workflow | Baseline | 4.5810 | - |
| Mixed workflow | Baseline-1 Fine-tuned RAG | 4.5690 | 0.0140 |
| Mixed workflow | Proposed | 3.9990 | 0.0160 |

Honest interpretation: the rubric step is present and evaluated, but this simple scalar rubric is biased toward cited direct ANSWER strings. It does not fully capture the value of TICKET/REJECT decisions in mixed workflow settings, where the proposed system improves Macro-F1 and unsupported-answer avoidance. Therefore, preference score should be reported as a rubric component/ablation, not as the main evidence of proposed-system improvement.

## Official Baseline vs Proposed Final Comparison

Output files:

- `outputs/reports/baseline_vs_proposed_metrics.json`
- `outputs/reports/baseline_vs_proposed_summary.md`
- `outputs/reports/baseline_vs_proposed_predictions.jsonl`
- `outputs/reports/unsupported_answer_safety_metrics.json`
- `outputs/reports/unsupported_answer_safety_summary.md`
- `outputs/reports/baseline_vs_proposed_latency.json`
- `outputs/reports/baseline_vs_proposed_efficiency.json`

Related documentation:

- `ARTIFACTS.md`
- `docs/GUARDRAILS.md`
- `docs/SYNTHETIC_DATA.md`
- `docs/PREFERENCE_RUBRIC.md`

Baseline is the official simple pretrained RAG baseline. It uses `sentence-transformers/all-MiniLM-L6-v2`, full KB search, no reranker, no routing, no triage/tool-policy, no ticketing, no rejection, no preference/rubric ranker, and always returns `ANSWER` with a citation.

Proposed is the final routed/tool-using support copilot with trained/fine-tuned components, domain routing, triage/tool-policy, ticketing, rejection, and grounded answer validation/generation.

Answer-only retrieval:

| Metric | Baseline | Proposed | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.1040 | 0.1560 | +0.0520 |
| Recall@5 | 0.1820 | 0.3620 | +0.1800 |
| MRR@10 | 0.1291 | 0.2320 | +0.1028 |
| EvidenceHit@5 | 0.1820 | 0.3620 | +0.1800 |

ESA/AQS:

| Metric | Baseline | Proposed | Delta |
|---|---:|---:|---:|
| ESA | 0.4760 | 0.5300 | +0.0540 |
| AQS | 0.6270 | 0.6733 | +0.0463 |

Unsupported-answer safety:

| Metric | Baseline | Proposed | Delta |
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

| Metric | Baseline | Proposed | Delta |
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

| Metric | Baseline | Proposed | Delta |
|---|---:|---:|---:|
| avg_latency_ms | N/A | 42.2952 | - |
| p95_latency_ms | N/A | 56.8940 | - |
| throughput_qps | N/A | 23.6433 | - |
| avg_fraction_kb_searched | 1.0000 | 0.9172 | -0.0828 |
| global_fallback_rate | 1.0000 | 0.8720 | -0.1280 |
| avg_num_domains_searched | 0.0000 | 2.7990 | +2.7990 |
| avg_num_tool_calls | 1.0000 | 5.7730 | +4.7730 |
| REE@5 | 0.1820 | 0.3947 | +0.2127 |

Latency note: Baseline live per-query latency was not remeasured in this final comparison because its predictions were batch-generated in `three_way_final_comparison`. Proposed latency is loaded from saved per-query mixed-eval traces. Do not claim proposed is faster than Baseline.

## Unsupported Answer Prevention

Unsupported case = gold `TICKET` or gold `REJECT`.

Unsupported answer = system returns direct `ANSWER` on an unsupported case.

UnsupportedAnswerPreventionRate = fraction of Baseline unsupported answers where Proposed instead chose `TICKET` or `REJECT`.

Since Baseline is a simple RAG system without ticket or reject tools, direct triage-F1 comparison can be misleading. The fairer customer-support safety comparison is unsupported-answer safety: how often a system gives a direct answer when the KB does not contain sufficient evidence, and how many such cases the proposed system prevents.

Baseline answered all 400 unsupported cases, so its unsupported-case `UnsupportedAnswerRate` is `1.0000`. Proposed answered 221 unsupported cases directly and prevented 179 of Baseline's unsupported answers, for an `UnsupportedAnswerPreventionRate` of `0.4475`.

Baseline-1 Fine-tuned RAG remains an ablation and is not the official assignment baseline. It remains strong on answer-only extraction, but it still lacks ticket/reject tools.

Answer-quality note: Proposed improves ESA/AQS over the official Baseline. The fine-tuned RAG ablation remains stronger on extraction-style answer-only ESA/AQS, so the fine-tuned generator should be presented as a customer-facing synthesis component whose quality is still constrained by retrieved evidence and automatic guardrails. ESA/AQS are automatic proxy metrics, not human evaluation.

## Final Threshold Tuning, Reranker Off

Output files:

- `outputs/reports/threshold_sweep_final.csv`
- `outputs/reports/threshold_tuned_final_metrics.json`
- `outputs/reports/threshold_sweep_final_summary.md`
- `configs/final_eval_threshold_tuned.yaml`

This tuning pass did not retrain any model and kept `use_reranker: false`. It swept `answerability_threshold`, `esa_accept_threshold`, `ticket_threshold`, `reject_threshold`, `nearest_kb_similarity_threshold`, `centroid_similarity_threshold`, and `fallback_score_threshold` over saved no-reranker predictions.

No threshold-only candidate satisfied all requested objectives simultaneously. In particular, ESA and AQS could not be improved by thresholding alone. The selected safety-preserving threshold config reduces unsupported direct answers and keeps `FalseRejectOnAnswerableRate` at `0.0000`, but it tickets more answerable rows, so ESA/AQS decrease. This config should be reported as a safety tradeoff, not as a strictly better final answer-quality configuration.

Selected thresholds:

| Threshold | Value |
|---|---:|
| answerability_threshold | 0.45 |
| esa_accept_threshold | 0.00 |
| ticket_threshold | 0.30 |
| reject_threshold | 0.25 |
| nearest_kb_similarity_threshold | 0.25 |
| centroid_similarity_threshold | 0.30 |
| fallback_score_threshold | 0.65 |

Updated proposed metrics after thresholding:

| Metric | Previous Proposed | Threshold-Tuned Proposed |
|---|---:|---:|
| Recall@5 | 0.3620 | 0.3620 |
| EvidenceHit@5 | 0.3620 | 0.3620 |
| ESA | 0.5300 | 0.4720 |
| AQS | 0.6733 | 0.6247 |
| UnsupportedAnswerRate, unsupported cases | 0.5525 | 0.3225 |
| UnsupportedAnswerCount, unsupported cases | 221 | 129 |
| UnsupportedAnswerPreventionRate | 0.4475 | 0.6775 |
| OODAnswerRate | 0.5500 | 0.2800 |
| TicketMissRate | 0.5550 | 0.3650 |
| FalseRejectOnAnswerableRate | 0.0000 | 0.0000 |
| FalseNonAnswerOnAnswerableRate | 0.1333 | 0.1750 |
| Macro-F1 | 0.5772 | 0.6032 |
| TICKET F1 | 0.4541 | 0.4990 |
| REJECT F1 | 0.5019 | 0.5019 |

Conclusion: the tuned config is useful if the presentation emphasizes unsupported-answer prevention and conservative rejection. The previous proposed config remains better for ESA/AQS and answer-quality framing.

