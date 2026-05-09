# Reject-Aware Domain-Routed Customer Support Copilot

This project implements a support RAG system over MultiDoc2Dial-style knowledge bases. It routes questions to likely support domains, searches cited KB passages, and uses structured tools to choose one of three support actions:

- `ANSWER`: return a grounded answer with citations.
- `TICKET`: create/escalate an account-specific or manual-review support issue.
- `REJECT`: conservatively reject clearly unsupported or out-of-domain queries.

The final system is designed to preserve answerable customer-question retrieval while adding workflow control and high-precision rejection.

## Problem Statement Mapping

- Retrieve KB passages with citations: yes.
- Tool calling: `RouteDomain`, `SearchKB`, `GetPolicy`, `CreateTicket`, `RejectQuery`.
- Escalation/ticket creation: yes.
- Baseline comparison: yes, against retrieval-only baseline RAG.
- Trained components: retriever, reranker, triage/tool-policy model, lightweight preference/rubric ranker.
- Demo: CLI and Streamlit UI.
- Latency: reported on the final mixed workflow evaluation.
- Structured tool schema: `schemas/tool_schema.json`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If Streamlit is not installed and you want the UI:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

## Data

The project uses IBM/MultiDoc2Dial-derived support documents, dialogue turns, and evidence references. Processed data is written under `data/processed/`.

The final workflow evaluation is mixed:

- ANSWER examples come from real MultiDoc2Dial answerable turns.
- TICKET examples are synthetic in-domain account-specific/manual-review support cases.
- REJECT examples are synthetic out-of-domain, support-like out-of-domain, and near-boundary queries.

This is intentional because MultiDoc2Dial mainly contains answerable dialogue turns and does not provide enough native ticket/reject workflow examples.

## Run Demo

CLI demo:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\final_eval_generator.yaml
```

Main final evaluation/demo config without generator fallback emphasis:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\final_eval_balanced_triage_best.yaml
```

Streamlit UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

See `DEMO_UI.md` for UI notes and presentation examples.

## Training Commands

Full pipeline training:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\train_retriever.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\build_index.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\train_reranker.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\train_preference.py --config configs\full.yaml
```

Balanced triage retraining used for the final workflow model:

```powershell
.\.venv\Scripts\python.exe scripts\build_balanced_triage_data.py
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\triage_balanced.yaml
```

Grounded generator fine-tuning:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\generator_finetune.yaml
.\.venv\Scripts\python.exe scripts\train_generator.py --config configs\generator_finetune.yaml
```

`prepare_data.py` now builds sentence-aware KB chunks and writes `data/processed/generator_train.jsonl`, `generator_val.jsonl`, and `generator_test.jsonl` from real user-to-agent dialogue turns when references are available.

The final cleanup did not retrain retriever, reranker, triage, or generator checkpoints.

## Evaluation Commands

Mixed workflow evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_mixed.py --config configs\final_eval_balanced_triage_best.yaml
```

Mixed grounding/evidence-use evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_grounding_mixed.py --config configs\final_eval_balanced_triage_best.yaml
```

Answer-only retrieval/grounding evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_answer_only.py --config configs\final_eval_generator.yaml
```

Answer-quality and formatting evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_answer_quality.py --config configs\final_eval_generator.yaml
```

## Final Results

### Table A: Mixed Workflow

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| Macro-F1 | 0.250 | 0.577 |
| Tool Decision Accuracy | 0.600 | 0.676 |
| ANSWER F1 | 0.750 | 0.776 |
| TICKET F1 | 0.000 | 0.454 |
| REJECT F1 | 0.000 | 0.502 |
| Reject Precision | 0.000 | 1.000 |
| FalseRejectRate | 0.000 | 0.000 |

### Table B: Evidence-Use Grounding

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| SupportedResponseRate | 0.575 | 0.658 |
| UnsupportedAnswerRate | 0.425 | 0.319 |
| EvidenceUseAccuracy | 0.600 | 0.676 |

### Table C: Answer-Only Retrieval/Grounding

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| Recall@5 | 0.360 | 0.362 |
| EvidenceHit@5 | 0.360 | 0.362 |
| CitationPrecision | 1.000 | 1.000 |

The fresh sentence-aware run slightly improves Recall@5 over the fresh baseline, but retrieval is still not the main gain; the main gain is workflow and evidence-use behavior.

### Table D: Answer Quality / Formatting

Reference answer text is now available for the fresh answer-only evaluation, so token F1 and ROUGE-L are reported. Formatting improves, but lexical overlap is lower than the baseline.

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| AnswerTokenF1 | 0.193 | 0.170 |
| ROUGE-L | 0.163 | 0.140 |
| NoFragmentRate | 0.838 | 1.000 |
| FragmentRate | 0.162 | 0.000 |
| DuplicateCitationRate | 1.000 | 0.000 |
| EmptyOrInvalidAnswerRate | 0.016 | 0.000 |
| CompleteAnswerRate | 0.838 | 1.000 |
| CitationAttachedRate | 1.000 | 1.000 |

### Latency

Final mixed workflow latency:

| Metric | Value |
|---|---:|
| Average latency | 49.88 ms |
| p95 latency | 62.80 ms |
| Throughput | 20.05 qps |

## Grounded Generator

`configs/final_eval_generator.yaml` enables a small grounded generator path for `ANSWER` synthesis. Tool decisions are still made by routing and triage/tool-policy logic. The generator receives the user question and retrieved evidence and is instructed to answer only from evidence or return `INSUFFICIENT_EVIDENCE`.

Citations are attached by the system from retrieved passages, not generated by the model. If FLAN-T5 is unavailable, returns insufficient evidence, or produces a fragment/citation-like output, inference falls back to the extractive synthesizer in `src/generation/extractive_synthesizer.py`.

For task-specific generator training, use `configs/generator_finetune.yaml`; the trained checkpoint is expected at `outputs/generator/flan_t5` and can be used with `configs/final_eval_generator_finetuned.yaml`.

## Important Limitations

- TICKET and REJECT examples are partly synthetic because MultiDoc2Dial mainly contains answerable turns.
- The final system preserves retrieval rather than significantly improving retrieval.
- Reject behavior is intentionally conservative to avoid rejecting answerable customer questions.
- Answer quality remains a limitation compared with a fully trained grounded LLM generator.
- No DPO or generator fine-tuning is claimed.
- FLAN-T5 should only be described as used when the model is actually available in the local environment; validated inference can still fall back to extractive synthesis when generation quality is insufficient.

## Earlier Calibration Runs / Ablations

Earlier debug and calibration runs produced weaker proposed retrieval and routing numbers. Those files are kept under `outputs/reports/archive/` for auditability, but they are not the final submitted results. The final narrative should use `outputs/reports/FINAL_RESULTS_FOR_REPORT.md` and the final metrics files listed in `outputs/reports/REPORT_INDEX.md`.
