# Metric Provenance

This project has several historical result files from calibration and ablation runs. For final submission, cite only the metric set below unless a table is explicitly labeled as an ablation.

## Headline Final Results

| Metric family | Source file | Config / setting | Notes |
|---|---|---|---|
| Answer-only retrieval | `outputs/reports/baseline0_vs_proposed_supported_synthesis_metrics.json` | Baseline-0: `configs/baseline_pretrained_rag.yaml`; Proposed: `configs/proposed_final.yaml` alias of the final generator/validation setting | Same retrieved hits/citations as the official comparison. |
| ESA/AQS | `outputs/reports/baseline0_vs_proposed_supported_synthesis_metrics.json` and `outputs/reports/esa_aqs_metrics.json` | Supported-synthesis answer text update | ESA/AQS changed only answer wording for proposed `ANSWER` rows; routing, retrieved hits, citations, tickets, and rejects are unchanged. |
| Mixed workflow / triage | `outputs/reports/baseline0_vs_proposed_supported_synthesis_metrics.json` | `data/processed/eval_mixed_1000.jsonl`, 600 ANSWER / 200 TICKET / 200 REJECT | Headline values: Tool Decision Accuracy `0.6760`, TICKET F1 `0.4541`, REJECT F1 `0.5019`, Macro-F1 `0.5772`. |
| Unsupported-answer safety | `outputs/reports/baseline0_vs_proposed_supported_synthesis_metrics.json` and `outputs/reports/unsupported_answer_safety_metrics.json` | Same mixed eval setting | UnsupportedAnswerRate over unsupported cases: Baseline-0 `1.0000`, Proposed `0.5525`. |
| Efficiency / tool usage | `outputs/reports/baseline0_vs_proposed_supported_synthesis_metrics.json` | Same mixed eval setting | Proposed latency is from saved traces; Baseline-0 live latency was not remeasured. |

## Archived Or Historical Runs

Files named `final_mixed_best_*`, `fresh_mixed_*`, `old_run_*`, and `three_way_final_comparison.*` are retained under `archive/old_reports/` for auditability. They are not the final submitted metric source unless a report section explicitly labels them as an ablation or historical run.

In the current repository, the archived `final_mixed_best_metrics.json` file reports the same mixed workflow headline values as the official comparison: Tool Decision Accuracy `0.6760`, TICKET F1 `0.4541`, REJECT F1 `0.5019`, and Macro-F1 `0.5772`. Older values from prior calibration runs should not be mixed into the final tables.

## Reporting Rule

Do not present multiple tables as “final” unless each table states:

- source file,
- config,
- evaluation set,
- whether reranker is on or off,
- whether supported-synthesis answer text is applied.
