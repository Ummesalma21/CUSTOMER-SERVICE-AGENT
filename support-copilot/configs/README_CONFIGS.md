# Config Guide

This folder contains only final or reproducible configs. Older smoke, debug, calibration, and failed-run configs are preserved in `archive/old_configs/`.

## Official Final Configs

- `baseline_pretrained_rag.yaml`: official Baseline-0. Uses pretrained `sentence-transformers/all-MiniLM-L6-v2`, full KB search, no routing, no reranker, no triage, no ticketing, no rejection.
- `proposed_final.yaml`: main final proposed system. Uses the trained retriever/index, domain routing, balanced triage/tool-policy, conservative ticket/reject behavior, and grounded answer validation/generation. Reranker is off.
- `safety_tuned_ablation.yaml`: threshold-tuned safety ablation. It reduces unsupported direct answers but lowers ESA/AQS, so it is not the main final answer-quality config.
- `reranker_ablation.yaml`: proposed-system reranker ablation. It is useful for analysis, but the final reported architecture keeps reranker off.

## Reproduction Configs

- `train_full.yaml`: full training/reproduction config alias for the final training intent.
- `triage_balanced.yaml`: balanced triage/tool-policy retraining config.
- `generator_fixed.yaml`: stable generator training/inference config used during generator debugging.

## Archive

Historical configs, including `smoke.yaml`, `debug_real.yaml`, older `final_eval_*` variants, failed generator configs, and rechunking experiments, live in `archive/old_configs/`.
