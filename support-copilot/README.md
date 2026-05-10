# Reject-Aware Domain-Routed Customer Support Copilot

## Overview

This project is a customer-support RAG system over a MultiDoc2Dial-style knowledge base. It answers support questions with citations, creates tickets when the issue is account-specific or lacks enough KB evidence, and rejects out-of-domain queries. The final comparison is against the official assignment baseline: a simple pretrained RAG system.

## Official Baseline

**Baseline** uses:

- `sentence-transformers/all-MiniLM-L6-v2` directly, with no fine-tuned retriever checkpoint
- full-KB dense search
- always `ANSWER` with a citation from retrieved evidence
- no reranker
- no domain routing
- no triage/tool-policy model
- no `CreateTicket` or `RejectQuery`
- no preference/rubric module

## Proposed System

The proposed system combines trained neural modules with lightweight answerability guardrails. Learned retrieval and triage/tool-policy models provide the main decision signals; rule-based checks are used only to prevent unsupported answers, vague-query failures, and malformed generations.

The proposed system includes:

- fine-tuned dense retriever
- trained DistilBERT `ANSWER` / `TICKET` / `REJECT` triage model
- domain routing and KB/domain similarity checks
- structured tools: `RouteDomain`, `SearchKB`, `GetPolicy`, `CreateTicket`, `RejectQuery`
- trained reranker, reported as an ablation/final tradeoff rather than the headline setting
- lightweight rubric-based preference/ranking module
- grounded answer validation/generation with system-attached citations

The decision policy is two-phase:

- **Phase 1: answerability guardrails.** Broad support-domain signals, account-specific patterns, KB proximity, centroid similarity, and evidence sufficiency checks catch high-confidence vague, unsupported, or account-specific cases.
- **Phase 2: learned and semantic decision.** Remaining queries use domain routing, dense retrieval, DistilBERT triage/tool-policy, evidence sufficiency, and answer validation to choose `ANSWER`, `TICKET`, or `REJECT`.

## Trained Components

The assignment requires training or fine-tuning at least three of retriever, reranker, generator, preference alignment, and tool-policy model. We trained/fine-tuned: (A) dense retriever, (B) cross-encoder reranker, (C) FLAN-T5-small grounded answer generator, and (E) DistilBERT tool-policy/triage model. We also implemented a lightweight rubric preference ranker.

| Assignment option | Component | Evidence/status |
|---|---|---|
| A Retriever | fine-tuned `sentence-transformers/all-MiniLM-L6-v2` | trained |
| B Reranker | fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2` | trained, reported as ablation/final tradeoff |
| C Generator | fine-tuned `google/flan-t5-small` | trained and used for answer synthesis when available |
| D Preference/rubric alignment | lightweight rubric response ranker | implemented, not DPO |
| E Tool-policy model | DistilBERT `ANSWER` / `TICKET` / `REJECT` classifier | trained |

The extractive synthesizer is kept as a fallback for environments where the local generator checkpoint is unavailable or returns insufficient evidence. Tool decisions and citations remain system-controlled.

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

## Demo

CLI:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\proposed.yaml
```

Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

More demo notes are in `DEMO_UI.md`.

## Training

Full reproduction:

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
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\train_triage_balanced.yaml
```

## Evaluation

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baseline_vs_proposed.py
.\.venv\Scripts\python.exe scripts\evaluate_esa_aqs.py
.\.venv\Scripts\python.exe scripts\evaluate_unsupported_answer_safety.py
```

Metric definitions are in `docs/METRICS.md`. Metric provenance is in `outputs/reports/METRIC_PROVENANCE.md`.

## Final Results

### Answer-Only Retrieval

| Metric | Baseline | Proposed |
|---|---:|---:|
| Recall@1 | 0.1040 | 0.1560 |
| Recall@5 | 0.1820 | 0.3620 |
| MRR@10 | 0.1291 | 0.2320 |
| EvidenceHit@5 | 0.1820 | 0.3620 |

### ESA/AQS

| Metric | Baseline | Proposed |
|---|---:|---:|
| ESA | 0.4760 | 0.5300 |
| AQS | 0.6270 | 0.6733 |

### Unsupported-Answer Safety

| Metric | Baseline | Proposed |
|---|---:|---:|
| UnsupportedAnswerRate | 1.0000 | 0.5525 |
| UnsupportedAnswerPreventionRate | 0.0000 | 0.4475 |
| SafeActionRate | 0.0000 | 0.4475 |
| OODAnswerRate | 1.0000 | 0.5500 |
| TicketMissRate | 1.0000 | 0.5550 |
| FalseRejectOnAnswerableRate | 0.0000 | 0.0000 |

### Mixed Workflow

| Metric | Baseline | Proposed |
|---|---:|---:|
| Tool Decision Accuracy | 0.6000 | 0.6760 |
| Macro-F1 | 0.2500 | 0.5772 |
| TICKET F1 | 0.0000 | 0.4541 |
| REJECT F1 | 0.0000 | 0.5019 |
| SupportedResponseRate | 0.5750 | 0.6580 |
| EvidenceUseAccuracy | 0.6000 | 0.6760 |
| REE@5 | 0.1820 | 0.3947 |

Compared with the official simple pretrained RAG baseline, the proposed system improves retrieval, evidence support, answer-quality proxy score, and unsupported-answer safety. Baseline-1 Fine-tuned RAG remains a strong answer-only ablation but lacks support workflow actions.

## Ablations

- **Baseline-1 Fine-tuned RAG**: strong on extractive answer-only ESA/AQS, but has no ticket/reject tools.
- **Reranker-enabled proposed**: improves mixed workflow/safety tradeoffs, but adds latency and changes answer-only evidence behavior.
- **Safety-tuned thresholds**: prevent more unsupported answers, with lower ESA/AQS and answer coverage.
- **Generator/rechunking**: the fine-tuned FLAN-T5-small generator is used for answer wording when available; quality is still limited by retrieved evidence and automatic guardrails.

## Limitations

- Synthetic `TICKET` and `REJECT` labels are used because MultiDoc2Dial mainly provides answerable document-grounded turns.
- ESA/AQS are automatic proxy metrics, not human judgment.
- Preference alignment is lightweight rubric ranking, not DPO.
- Rule-based guardrails are used for answerability/safety; they are not demo-specific hardcoding.
- Fine-tuned RAG remains stronger on extractive answer-only ESA/AQS.
- Reranker and safety-threshold variants expose tradeoffs between answer quality, latency, and safety.

## Repository Layout

- `configs/`: final baseline, proposed, ablation, and training configs
- `scripts/`: training, evaluation, and demo scripts
- `src/`: retrieval, routing, triage, generation, tools, and policy modules
- `schemas/`: JSON tool schemas
- `docs/`: metrics, guardrails, synthetic data, and preference/rubric notes
- `outputs/reports/`: final report-facing metrics and summaries
- `report/` and `slides/`: submission report/slide notes
