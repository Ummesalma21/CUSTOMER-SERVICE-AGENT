# Final Results For Report

## Problem Statement

Build a reject-aware customer-support RAG system that can:

- answer supported KB questions with evidence,
- escalate account-specific/manual-review issues to `TICKET`,
- reject unsupported/out-of-domain requests.

For final submission, we report only two main systems:

1. **Baseline**
2. **Proposed: Domain-Router RAG with validation-tuned fallback**

## Baseline

Baseline is a simple full-KB retrieval pipeline with pretrained embeddings and no fallback tuning policy optimization.

Main baseline metrics are read from:

- `outputs/reports/baseline_vs_proposed_metrics.json`  
  (`answer_only.baseline`, `mixed_workflow.baseline`, `unsupported_answer_safety.baseline`)

## Proposed Domain-Router Method

The proposed system uses:

- a learned 4-domain router (`SSA`, `VA`, `SA`, `FSA`) to choose where to search first,
- domain-first retrieval over top-k routed domains,
- a validation-tuned fallback merge to global candidates when routed evidence is weak,
- a trained MLP action head for workflow/tool decision (`ANSWER` / `REJECT` / `TICKET`).

Pipeline (kept unchanged):

`query -> learned router -> top-k domain retrieval -> fallback thresholds -> optional global merge -> evidence selection -> trained MLP action head -> final workflow action`

Final proposed metrics are read from:

- `outputs/reports/domain_router_fallback_grid_fixed_metrics.json`
- `outputs/reports/domain_router_fallback_grid_fixed_summary.md`

## Validation-Tuned Fallback (Selected Setting)

- `top_k_domains=2`
- `min_domain_confidence=0.7`
- `min_candidate_similarity=0.4`
- `min_domain_candidates=5`
- `fallback_merge_mode=merge`
- `rerank_after_merge=true`

## Clean Held-Out Final Comparison

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
| Avg. domains searched | 2.0 | Router uses top-2 domain search instead of full-KB-first retrieval |
| Gold domain in searched domains | 0.685 | Top-2 gold-domain-in-searched coverage is 68.5% |
| Avg. domain confidence | 0.7828 | Router predictions are generally confident |
| Fallback trigger count | 799 | Fallback is frequently used to recover from weak routed evidence |

The routing diagnostics show that final gains come from a learned domain-first policy plus fallback recovery, not from scanning the full KB first.

## Exported Final Artifacts

- `outputs/final/baseline_final_metrics.json`
- `outputs/final/proposed_domain_router_final_metrics.json`
- `outputs/final/final_comparison_table.md`

## Limitations

- Mixed workflow evaluation includes synthetic `TICKET`/`REJECT` examples because MultiDoc2Dial is mostly answerable.
- ESA/AQS are automatic proxy metrics, not human preference studies.
- Domain-router quality depends on retrieval quality and fallback threshold calibration.

## Ablations / Archive Note

The following are retained as appendix/ablation artifacts only and are not headline final systems:

- neural cluster gate experiments,
- evidence-supervised router leaked/invalid runs,
- pre-fixed domain-router runs,
- domain-router + learned reranker run,
- old frozen proposed comparisons.
