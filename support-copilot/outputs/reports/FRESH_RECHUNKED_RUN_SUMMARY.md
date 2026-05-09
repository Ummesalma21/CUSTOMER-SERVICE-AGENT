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
