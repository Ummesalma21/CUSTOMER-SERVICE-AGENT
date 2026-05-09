# Reject-Aware Domain-Routed Customer Support Copilot

## Overview

This project builds a customer-support RAG system over MultiDoc2Dial-style support knowledge bases. The system retrieves cited KB evidence, routes questions to support domains, validates answerability, and chooses one support action: `ANSWER`, `TICKET`, or `REJECT`. The final claim is deliberately narrow: compared with a simple pretrained RAG baseline, the proposed system preserves answer retrieval while improving support workflow control and unsupported-answer safety.

## Final Systems

- **Baseline-0 Pretrained RAG**: official assignment baseline in `configs/baseline_pretrained_rag.yaml`. It uses pretrained `sentence-transformers/all-MiniLM-L6-v2`, full KB search, no reranker, no routing, no triage/tool-policy, no ticketing, no rejection, and always returns `ANSWER` with a citation.
- **Proposed Final System**: `configs/proposed_final.yaml`. It uses the trained retriever/index, domain routing, balanced triage/tool-policy, structured tools, conservative ticket/reject behavior, and grounded answer validation/generation. Reranker is off.
- **Baseline-1 Fine-tuned RAG**: an ablation only. It uses the fine-tuned retriever but has no workflow tools.
- **Safety-tuned and reranker variants**: ablations in `configs/safety_tuned_ablation.yaml` and `configs/reranker_ablation.yaml`.

Structured tool calls are specified in `schemas/tool_schema.json`.

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

## Ablations

- **Safety tuned**: prevents more unsupported answers, but lowers ESA/AQS and tickets more answerable cases.
- **Reranker**: improves some mixed workflow safety metrics, but hurts answer-only evidence quality and adds latency.
- **Fine-tuned RAG**: strong answer-only extraction ablation, but lacks ticket/reject workflow control.
- **Generator**: answer quality remains a limitation; supported synthesis improves ESA/AQS, but this is not a substitute for a fully trained grounded generator.

## Limitations

- TICKET and REJECT examples are partly synthetic because MultiDoc2Dial mainly contains answerable turns.
- The proposed system improves workflow and safety more than retrieval quality.
- Reject behavior is intentionally conservative to avoid rejecting answerable support questions.
- ESA/AQS are automatic proxy metrics, not human evaluation.
- The archive contains older failed or intermediate experiments for auditability, but they are not the final submitted results.
