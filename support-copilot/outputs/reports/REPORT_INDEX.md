# Report Index

## Current Fresh Run

- `FINAL_RESULTS_FOR_REPORT.md`: fresh rechunked full-run summary.
- `FRESH_RECHUNKED_RUN_SUMMARY.md`: same fresh-run summary with explicit archive/run details.
- `final_mixed_best_metrics.json`: current fresh mixed workflow metrics.
- `final_mixed_best_summary.md`: current fresh mixed workflow summary.
- `final_mixed_grounding_metrics.json`: current fresh grounding/evidence-use metrics.
- `final_mixed_grounding_summary.md`: current fresh grounding/evidence-use summary.
- `final_answer_only_metrics.json`: current fresh answer-only retrieval/grounding metrics.
- `final_answer_only_summary.md`: current fresh answer-only retrieval/grounding summary.
- `final_answer_quality_metrics.json`: current fresh answer-quality metrics.
- `final_answer_quality_summary.md`: current fresh answer-quality summary.
- `generator_train_metrics.json`: current generator fine-tuning metrics.
- `triage_balanced_dataset_summary.json`: balanced triage split/leakage summary.
- `triage_balanced_train_metrics.json`: balanced triage training metrics.
- `demo_quality_checks.md`: current fresh demo checks.

## Local Archive

The previous run was moved to `outputs/archive_runs/20260508_225006`. That directory contains old processed data, indexes, reports, logs, and checkpoints. It is ignored by git because it contains large local artifacts.

## Notes

Large prediction JSONL files, checkpoints, indexes, and logs are intentionally ignored by git. Use the metric JSON and Markdown summaries above for reporting.
