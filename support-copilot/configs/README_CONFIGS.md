# Config Guide

Use these configs for final submission:

- `final_eval_balanced_triage_best.yaml`: main final evaluation/demo config. Uses existing retriever/reranker/index artifacts and the balanced triage checkpoint with conservative reject thresholds.
- `final_eval_generator.yaml`: presentation/demo config that enables grounded answer synthesis. It tries local FLAN-T5 and falls back to extractive synthesis when the generator is unavailable.
- `generator_finetune.yaml`: supervised grounded-generator fine-tuning config using `data/processed/generator_train.jsonl`.
- `final_eval_generator_finetuned.yaml`: demo/eval config that uses `outputs/generator/flan_t5` after generator fine-tuning. If that checkpoint is missing, use `final_eval_generator.yaml`.
- `triage_balanced.yaml`: balanced triage retraining config for the final tool-policy checkpoint.
- `full.yaml`: intended full training config for data prep, retriever, index, reranker, triage, and preference/rubric ranker.
- `full_local.yaml`: reduced local training/development config.
- `smoke.yaml`: tiny scaffold/smoke test config.
- `debug_real.yaml`: small real-data debugging config.
- `final_eval_balanced_triage.yaml`, `final_eval_calibrated.yaml`, and `final_eval_mixed_best.yaml`: historical calibration/development configs kept for reproducibility.

The final reported metrics should be read from `outputs/reports/FINAL_RESULTS_FOR_REPORT.md` and the final metric files listed in `outputs/reports/REPORT_INDEX.md`.
