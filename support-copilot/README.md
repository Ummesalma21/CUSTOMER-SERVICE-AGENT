# Reject-Aware Domain-Routed Customer Support Copilot

## Overview

This project builds a customer-support RAG system over MultiDoc2Dial-style support knowledge bases. The system retrieves cited KB evidence, routes questions to support domains, validates answerability, and chooses one support action: `ANSWER`, `TICKET`, or `REJECT`. The final claim is deliberately narrow: compared with a simple pretrained RAG baseline, the proposed system preserves answer retrieval while improving support workflow control and unsupported-answer safety.

## Proposed Method

The proposed system uses a **two-phase reject-aware policy**.

**Phase 1: lexical safety gate.** A lightweight, interpretable gate catches high-confidence cases before expensive generation: obvious out-of-domain requests, vague unsupported questions, and account-specific/manual-review support issues. This phase is conservative and is not meant to reject answerable support questions.

**Phase 2: learned and semantic decision.** Queries that pass Phase 1 use domain routing, nearest-KB similarity, domain centroid similarity, a trained DistilBERT triage/tool-policy model, evidence sufficiency checks, and answer-quality validation. This phase chooses `ANSWER`, `TICKET`, or `REJECT` using retrieved evidence and learned support-workflow signals.

The design is hybrid by intent: Phase 1 provides transparent safety guardrails, while Phase 2 provides the learned retrieval, routing, and triage behavior.

## Final Systems

- **Baseline-0 Pretrained RAG**: official assignment baseline in `configs/baseline_pretrained_rag.yaml`. It uses pretrained `sentence-transformers/all-MiniLM-L6-v2`, full KB search, no reranker, no routing, no triage/tool-policy, no ticketing, no rejection, and always returns `ANSWER` with a citation.
- **Proposed Final System**: `configs/proposed_final.yaml`. It uses the trained retriever/index, domain routing, balanced triage/tool-policy, structured tools, conservative ticket/reject behavior, and grounded answer validation/generation. Reranker is off.
- **Baseline-1 Fine-tuned RAG**: an ablation only. It uses the fine-tuned retriever but has no workflow tools.
- **Safety-tuned and reranker variants**: ablations in `configs/safety_tuned_ablation.yaml` and `configs/reranker_ablation.yaml`.

Structured tool calls are specified in `schemas/tool_schema.json`.

Additional submission notes:

- artifact and regeneration path: `ARTIFACTS.md`
- answerability/safety guardrails: `docs/GUARDRAILS.md`
- metric definitions: `docs/METRICS.md`
- synthetic TICKET/REJECT data: `docs/SYNTHETIC_DATA.md`
- preference/rubric module: `docs/PREFERENCE_RUBRIC.md`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional UI dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

## Demo Commands

CLI:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\proposed_final.yaml
```

Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

More demo notes are in `DEMO_UI.md`.

## Training And Reproduction Commands

Full pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_retriever.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\build_index.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_reranker.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_preference.py --config configs\train_full.yaml
```

Balanced triage/tool-policy:

```powershell
.\.venv\Scripts\python.exe scripts\build_balanced_triage_data.py
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\triage_balanced.yaml
```

Generator debugging/fixed generator:

```powershell
.\.venv\Scripts\python.exe scripts\train_generator.py --config configs\generator_fixed.yaml
```

## Evaluation Commands

Official Baseline-0 vs proposed comparison:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baseline0_vs_proposed.py
```

ESA/AQS:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_esa_aqs.py
```

Unsupported-answer safety:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_unsupported_answer_safety.py
```

Threshold ablation:

```powershell
.\.venv\Scripts\python.exe scripts\tune_final_thresholds.py
```

## Final Results

Headline metrics are reported from one final setting: Baseline-0 `configs/baseline_pretrained_rag.yaml` versus Proposed `configs/proposed_final.yaml`, reranker off, evaluated on `data/processed/eval_mixed_1000.jsonl` and the answer-only evaluation set. The source files are `outputs/reports/baseline0_vs_proposed_supported_synthesis_metrics.json`, `outputs/reports/esa_aqs_metrics.json`, and `outputs/reports/METRIC_PROVENANCE.md`.

Metric definitions are in `docs/METRICS.md`.

| Metric | Baseline-0 Pretrained RAG | Proposed Final |
|---|---:|---:|
| Recall@5 | 0.1820 | 0.3620 |
| EvidenceHit@5 | 0.1820 | 0.3620 |
| ESA | 0.4760 | 0.6380 |
| AQS | 0.6270 | 0.7187 |
| UnsupportedAnswerRate | 1.0000 | 0.5525 |
| UnsupportedAnswerPreventionRate | 0.0000 | 0.4475 |
| Macro-F1 | 0.2500 | 0.5772 |
| REE@5 | 0.1820 | 0.3947 |

Final report files live in `outputs/reports/`; start with `outputs/reports/FINAL_RESULTS_FOR_REPORT.md` and `outputs/reports/REPORT_INDEX.md`.

Metric hygiene note: older files with names such as `final_mixed_best_*`, `fresh_mixed_*`, `old_run_*`, and `three_way_final_comparison.*` are archived historical runs. They are not the headline final results unless explicitly labeled as an ablation. See `outputs/reports/METRIC_PROVENANCE.md`.

The proposed system improves ESA/AQS over the official Baseline-0 pretrained RAG baseline. The fine-tuned RAG ablation remains stronger on extraction-style answer-only ESA/AQS, so grounded generation should still be treated as a limitation. ESA/AQS are automatic proxy metrics, not human evaluation.

## Ablations

- **Safety tuned**: trades answer-quality proxy score for stronger unsupported-answer prevention.
- **Reranker**: evaluated as a separate workflow-safety and latency ablation.
- **Fine-tuned RAG**: strong answer-only extraction ablation, but lacks ticket/reject workflow control.
- **Generator / synthesis**: used as a bounded answer-formulation layer; supported synthesis improves ESA/AQS while citations remain system-attached.

## Limitations

- TICKET and REJECT examples are partly synthetic because MultiDoc2Dial mainly contains answerable turns.
- The proposed system improves workflow and safety more than retrieval quality.
- Reject behavior is intentionally conservative to avoid rejecting answerable support questions.
- ESA/AQS are automatic proxy metrics, not human evaluation.
- The archive contains intermediate experiments for auditability, but they are not the final submitted results.
