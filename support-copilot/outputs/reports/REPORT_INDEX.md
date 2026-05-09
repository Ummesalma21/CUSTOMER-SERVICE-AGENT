# Report Index

## Main Final Reports

- `FINAL_RESULTS_FOR_REPORT.md`: complete final narrative, tables, models, limitations, and submission-ready interpretation.
- `METRIC_PROVENANCE.md`: authoritative mapping from headline metric values to source files, configs, and evaluation settings.
- `baseline0_vs_proposed_summary.md`: official Baseline-0 pretrained RAG vs proposed system comparison, updated to cite the supported-synthesis ESA/AQS values.
- `baseline0_vs_proposed_metrics.json`: pre-supported-synthesis machine-readable comparison; keep for audit, but use the supported-synthesis file for headline ESA/AQS.
- `baseline0_vs_proposed_supported_synthesis_metrics.json`: headline machine-readable final comparison with the supported-synthesis ESA/AQS update.
- `supported_synthesis_answer_improvement_summary.md`: confirms supported-synthesis answer rewriting changed answer text only, not decisions or retrieved hits.
- `esa_aqs_summary.md`: ESA and AQS evidence-support/answer-quality proxy summary.
- `esa_aqs_metrics.json`: machine-readable ESA/AQS metrics.
- `unsupported_answer_safety_summary.md`: unsupported-answer safety framing for Baseline-0 vs proposed.
- `unsupported_answer_safety_metrics.json`: machine-readable unsupported-answer safety metrics.

## Ablation Reports

- `threshold_sweep_final_summary.md`: threshold-only safety tuning summary.
- `threshold_tuned_final_metrics.json`: selected safety-tuned ablation metrics.
- `reranker_ablation_summary.md`: reranker-enabled proposed-system ablation summary.
- `generator_debug_final_summary.md`: generator debugging and fixed-generator status.
- `preference_score_comparison.md`: lightweight preference/rubric ranker score comparison.
- `preference_score_comparison.json`: machine-readable preference/rubric score comparison.

## Demo

- `demo_quality_checks.md`: final CLI demo checks and expected behavior.

## Archive Locations

- `archive/old_reports/`: old summaries, duplicate metrics, debug reports, and non-final markdown/json reports.
- `archive/old_outputs/`: bulky prediction dumps, CSV sweeps, and per-row scored outputs.
- `archive/old_logs/`: logs from training, debugging, and older evaluations.
- `archive/old_runs/`: old full run snapshots.

Archived files with names such as `final_mixed_best_*`, `fresh_mixed_*`, `old_run_*`, and `three_way_final_comparison.*` are retained for auditability. They should not be cited as final headline metrics unless a report section explicitly labels the config and evaluation setting.

## Supporting Documentation

- `ARTIFACTS.md`: included/excluded artifacts and regeneration commands.
- `docs/GUARDRAILS.md`: answerability and safety guardrail policy.
- `docs/METRICS.md`: definitions for retrieval, workflow, unsupported-answer safety, ESA/AQS, and efficiency metrics.
- `docs/SYNTHETIC_DATA.md`: synthetic TICKET/REJECT data rationale and limitations.
- `docs/PREFERENCE_RUBRIC.md`: lightweight preference/rubric module description.
