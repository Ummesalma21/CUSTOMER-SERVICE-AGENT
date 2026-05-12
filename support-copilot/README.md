# Reject-Aware Domain-Routed Customer Support Copilot

## Overview

This project is a support RAG system over a MultiDoc2Dial-style KB. It answers supported questions with citations, creates tickets for account-specific/manual-review issues, and rejects unsupported queries.  
For final submission, we report only two main systems:

1. **Baseline**
2. **Proposed: Domain-Router RAG with validation-tuned fallback**

---

## Main Systems

### Baseline

- Pretrained retriever (`sentence-transformers/all-MiniLM-L6-v2`)
- Full-KB search
- Always returns `ANSWER`
- No domain router fallback tuning

### Proposed (Final)

- 4-domain router (`SSA`, `VA`, `SA`, `FSA`)
- Trained MLP action head for workflow/tool decision: `ANSWER` / `REJECT` / `TICKET`
- Domain-first retrieval with validation-tuned fallback merge
- Clean held-out evaluation
- Final selected thresholds:
  - `top_k_domains=2`
  - `min_domain_confidence=0.7`
  - `min_candidate_similarity=0.4`
  - `min_domain_candidates=5`
  - `fallback_merge_mode=merge`
  - `rerank_after_merge=true`

Proposed architecture (kept unchanged):

`query + short history -> learned domain router -> top-k domain retrieval -> fallback threshold check -> optional global merge -> evidence selection -> trained MLP action head -> ANSWER/REJECT/TICKET`

---

## Final Baseline vs Proposed Results

| Metric | Baseline | Proposed (Domain-Router) |
|---|---:|---:|
| Recall@5 | 0.1820 | 0.3473 |
| EvidenceHit@5 | 0.1820 | 0.3473 |
| ESA | 0.4760 | 0.7936 |
| AQS | 0.6270 | 0.8195 |
| Macro-F1 | 0.2500 | 0.8257 |
| UnsupportedAnswerRate | 1.0000 | 0.2250 |
| OODAnswerRate | 1.0000 | 0.4500 |
| TicketMissRate | 1.0000 | 0.0000 |

## Routing Diagnostics (Proposed)

| Diagnostic | Value | Interpretation |
|---|---:|---|
| Avg. domains searched | 2.0 | Router uses top-2 domain search |
| Gold domain in searched domains | 0.685 | Gold-domain-in-searched coverage for top-2 routing |
| Avg. domain confidence | 0.7828 | Router predictions are generally confident |
| Fallback trigger count | 799 | Fallback is actively used to protect recall |

These diagnostics show a meaningful domain-first search policy with fallback recovery, not full-KB-first retrieval.

Final exported files:

- `outputs/final/baseline_final_metrics.json`
- `outputs/final/proposed_domain_router_final_metrics.json`
- `outputs/final/final_comparison_table.md`

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduction

### 1) Prepare data/index (if missing)

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\build_index.py --config configs\train_full.yaml
```

### 2) Train domain router and tune fallback

```powershell
.\.venv\Scripts\python.exe scripts\prepare_domain_router_clean_split.py
.\.venv\Scripts\python.exe scripts\train_domain_router.py --config configs\domain_router_experiment.yaml
.\.venv\Scripts\python.exe scripts\tune_domain_router_fallback_grid_fixed.py --config configs\domain_router_experiment.yaml
```

### 3) Export final comparison artifacts

```powershell
.\.venv\Scripts\python.exe scripts\export_final_results.py
```

---

## Limitations

- Mixed workflow includes synthetic `TICKET`/`REJECT` examples because MultiDoc2Dial is mostly answerable.
- ESA/AQS are automatic proxy metrics.
- Ablation runs are kept for auditability but are not part of the headline final comparison.

---

## Ablations / Archive Note

The following remain as appendix/ablation-only analyses (not headline results):

- Neural cluster gate experiments
- Evidence-supervised router leaked/invalid runs
- Pre-fixed domain-router runs
- Learned-reranker domain-router run
- Older frozen proposed comparisons

See `outputs/reports/` for those records.
