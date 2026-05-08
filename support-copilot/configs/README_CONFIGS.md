# Config Guide

Use these configs for final submission:

- `final_eval_balanced_triage_best.yaml`: main final evaluation/demo config. Uses existing retriever/reranker/index artifacts and the balanced triage checkpoint with conservative reject thresholds.
- `final_eval_generator.yaml`: presentation/demo config that enables grounded answer synthesis. It tries local FLAN-T5 and falls back to extractive synthesis when the generator is unavailable.
- `triage_balanced.yaml`: balanced triage retraining config for the final tool-policy checkpoint.
- `full.yaml`: intended full training config for data prep, retriever, index, reranker, triage, and preference/rubric ranker.
- `full_local.yaml`: reduced local training/development config.
- `smoke.yaml`: tiny scaffold/smoke test config.
- `debug_real.yaml`: small real-data debugging config.
- `final_eval_balanced_triage.yaml`, `final_eval_calibrated.yaml`, and `final_eval_mixed_best.yaml`: historical calibration/development configs kept for reproducibility.

The final reported metrics should be read from `outputs/reports/FINAL_RESULTS_FOR_REPORT.md` and the final metric files listed in `outputs/reports/REPORT_INDEX.md`.
