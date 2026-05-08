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

The final cleanup did not retrain retriever, reranker, or triage checkpoints.

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
| Macro-F1 | 0.250 | 0.674 |
| Tool Decision Accuracy | 0.600 | 0.765 |
| ANSWER F1 | 0.750 | 0.835 |
| TICKET F1 | 0.000 | 0.614 |
| REJECT F1 | 0.000 | 0.574 |
| Reject Precision | 0.000 | 0.933 |
| FalseRejectRate | 0.000 | 0.0075 |

### Table B: Evidence-Use Grounding

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| SupportedResponseRate | 0.551 | 0.718 |
| UnsupportedAnswerRate | 0.449 | 0.282 |
| EvidenceUseAccuracy | 0.600 | 0.765 |

### Table C: Answer-Only Retrieval/Grounding

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| Recall@5 | 0.4520 | 0.4513 |
| EvidenceHit@5 | 0.4520 | 0.4513 |
| CitationPrecision | 1.000 | 1.000 |

The proposed system preserves near-baseline answer retrieval; it does not significantly improve retrieval over the baseline.

### Table D: Answer Quality / Formatting

Reference answer text was not available in the answer-only prediction file, so token F1 and ROUGE-L are not reported. The following no-reference formatting and citation metrics are computed over answerable outputs.

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| NoFragmentRate | 0.697 | 1.000 |
| FragmentRate | 0.303 | 0.000 |
| DuplicateCitationRate | 1.000 | 0.000 |
| EmptyOrInvalidAnswerRate | 0.091 | 0.000 |
| CompleteAnswerRate | 0.697 | 1.000 |
| CitationAttachedRate | 1.000 | 1.000 |

### Latency

Final mixed workflow latency:

| Metric | Value |
|---|---:|
| Average latency | 99.28 ms |
| p95 latency | 145.20 ms |
| Throughput | 10.07 qps |

## Grounded Generator

`configs/final_eval_generator.yaml` enables a small grounded generator path for `ANSWER` synthesis. Tool decisions are still made by routing and triage/tool-policy logic. The generator receives the user question and retrieved evidence and is instructed to answer only from evidence or return `INSUFFICIENT_EVIDENCE`.

Citations are attached by the system from retrieved passages, not generated by the model. If FLAN-T5 is unavailable, returns insufficient evidence, or produces a fragment/citation-like output, inference falls back to the extractive synthesizer in `src/generation/extractive_synthesizer.py`.

## Important Limitations

- TICKET and REJECT examples are partly synthetic because MultiDoc2Dial mainly contains answerable turns.
- The final system preserves retrieval rather than significantly improving retrieval.
- Reject behavior is intentionally conservative to avoid rejecting answerable customer questions.
- Answer quality remains a limitation compared with a fully trained grounded LLM generator.
- No DPO or generator fine-tuning is claimed.
- FLAN-T5 should only be described as used when the model is actually available in the local environment; validated inference can still fall back to extractive synthesis when generation quality is insufficient.

## Earlier Calibration Runs / Ablations

Earlier debug and calibration runs produced weaker proposed retrieval and routing numbers. Those files are kept under `outputs/reports/archive/` for auditability, but they are not the final submitted results. The final narrative should use `outputs/reports/FINAL_RESULTS_FOR_REPORT.md` and the final metrics files listed in `outputs/reports/REPORT_INDEX.md`.
