# Report Index

## Final Result Files

- `FINAL_RESULTS_FOR_REPORT.md`: main final results narrative for the course report.
- `final_mixed_best_metrics.json`: final mixed workflow metrics, including baseline and proposed comparisons.
- `final_mixed_best_summary.md`: final mixed workflow summary.
- `final_mixed_grounding_metrics.json`: mixed workflow grounding/evidence-use metrics.
- `final_mixed_grounding_summary.md`: grounding/evidence-use summary.
- `final_answer_only_metrics.json`: answer-only retrieval and grounding metrics.
- `final_answer_only_summary.md`: answer-only retrieval and grounding summary.
- `final_answer_quality_metrics.json`: answer-quality and formatting metrics.
- `final_answer_quality_summary.md`: answer-quality summary.
- `demo_quality_checks.md`: CLI demo verification traces.
- `latency_metrics.json`: final mixed workflow latency copied from `final_mixed_best_metrics.json`.

## Training And Data Summaries

- `data_stats.json`: final full-data run sizes.
- `triage_balanced_dataset_summary.json`: balanced triage dataset class counts and leakage check.
- `triage_balanced_train_metrics.json`: balanced triage training metrics.
- `ablation_metrics.csv`: retained ablation table from the final experiment run.

## Large Local Outputs

Prediction JSONL files such as `final_mixed_best_predictions.jsonl`, `final_answer_only_predictions.jsonl`, and `final_answer_quality_predictions_scored.jsonl` may exist locally. They are ignored by git because they are large, but the metric summaries above are retained.

## Archived / Intermediate Files

Old debug, calibration, and earlier failed-result artifacts have been moved to `outputs/reports/archive/`. They are kept for auditability and should not be presented as final submitted metrics.

Archived examples include earlier `proposed_metrics.json`, older mixed-evaluation files, threshold sweeps, old `final_summary.md`, and large intermediate prediction/tool-trace dumps.

## Logs

Runtime logs are written to `outputs/logs/`. Large `*.log` files are ignored by git; use the report JSON/Markdown files above for submitted results.
