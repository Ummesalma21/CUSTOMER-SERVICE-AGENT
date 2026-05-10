# Report Index

## Main Final Reports

- `FINAL_RESULTS_FOR_REPORT.md`: complete final narrative, tables, models, limitations, and submission-ready interpretation.
- `METRIC_PROVENANCE.md`: authoritative mapping from headline metric values to source files, configs, and evaluation settings.
- `baseline_vs_proposed_summary.md`: official Baseline vs proposed system comparison.
- `baseline_vs_proposed_metrics.json`: machine-readable official final comparison.
- `esa_aqs_summary.md`: ESA and AQS evidence-support/answer-quality proxy summary.
- `esa_aqs_metrics.json`: machine-readable ESA/AQS metrics.
- `unsupported_answer_safety_summary.md`: unsupported-answer safety framing for Baseline vs proposed.
- `unsupported_answer_safety_metrics.json`: machine-readable unsupported-answer safety metrics.

## Supporting Reports

- `generator_debug_final_summary.md`: generator and answer synthesis summary.
- `preference_score_comparison.md`: lightweight preference/rubric ranker score comparison.
- `preference_score_comparison.json`: machine-readable preference/rubric score comparison.

## Demo

- `demo_quality_checks.md`: final CLI demo checks and expected behavior.

## Archive Locations

- local `archive/old_reports/`: old summaries, duplicate metrics, debug reports, and non-final markdown/json reports. This folder is excluded from Git/ZIP.
- `archive/old_outputs/`: bulky prediction dumps, CSV sweeps, and per-row scored outputs.
- `archive/old_logs/`: logs from training, debugging, and older evaluations.
- `archive/old_runs/`: old full run snapshots.

Archived files with names such as `final_mixed_best_*`, `fresh_mixed_*`, `old_run_*`, ablation sweeps, and `three_way_final_comparison.*` are retained locally for auditability. They are excluded from Git/ZIP and should not be cited as final headline metrics unless explicitly restored and labeled.

## Supporting Documentation

- `ARTIFACTS.md`: included/excluded artifacts and regeneration commands.
- `docs/GUARDRAILS.md`: answerability and safety guardrail policy.
- `docs/METRICS.md`: definitions for retrieval, workflow, unsupported-answer safety, ESA/AQS, and efficiency metrics.
- `docs/SYNTHETIC_DATA.md`: synthetic TICKET/REJECT data rationale and limitations.
- `docs/PREFERENCE_RUBRIC.md`: lightweight preference/rubric module description.
