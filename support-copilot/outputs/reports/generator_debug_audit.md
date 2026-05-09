# Generator Debug Audit

## Files Inspected

- Training CLI: `scripts/train_generator.py`
- Training implementation: `src/generation/train_generator_lora.py`
- Dataset creation: `src/data/make_generator_data.py`, called from `src/data/preprocess.py`
- Inference module: `src/generation/grounded_generator.py`
- Extractive fallback: `src/generation/extractive_synthesizer.py`
- Answer validation: `src/generation/answer_quality.py`
- Answer-quality evaluation: `scripts/evaluate_answer_quality.py`
- Full training config: `configs/fresh_rechunked_full.yaml`
- Previous generator inference config: `configs/final_eval_generator_finetuned.yaml`
- Fixed full config: `configs/generator_fixed_full.yaml`

## Dataset Paths

- Train: `data/processed/generator_train.jsonl`
- Validation: `data/processed/generator_val.jsonl`
- Test: `data/processed/generator_test.jsonl`

Observed sizes:

- Train: `10080`
- Validation: `960`
- Test: `960`

## Checkpoints

- Previous unstable checkpoint: `outputs/generator/flan_t5`
- Fixed checkpoint target: `outputs/generator/flan_t5_fixed`

The previous checkpoint exists locally. It is not overwritten by the fixed training path.

## Current Generator Config

Previous fresh run used:

- Model: `google/flan-t5-small`
- Output: `outputs/generator/flan_t5`
- Batch size: `2`
- Gradient accumulation: `8`
- Epochs: `1`
- FP16: `true`
- Max source length: `768`
- Max target length: `128`

Fixed config uses:

- Model: `google/flan-t5-small`
- Output: `outputs/generator/flan_t5_fixed`
- Batch size: `2`
- Gradient accumulation: `8`
- Epochs: `1`
- FP16: `false`
- Max source length: `512`
- Max target length: `96`
- Gradient clipping: `1.0`

## Inference Path

`src/generation/grounded_generator.py` attempts the configured generator first. If the model is unavailable, emits `INSUFFICIENT_EVIDENCE`, or produces fragmentary output, it falls back to `src/generation/extractive_synthesizer.py`.

The CLI/UI display goes through answer-quality postprocessing, so invalid generator text may be replaced by a ticket/reject/clarification path or by extractive synthesis.

## Losses And Metrics

Previous unstable metrics were written to `outputs/reports/generator_train_metrics.json`:

- Train loss: `0.0`
- Eval loss: `null` after NaN cleanup
- Eval perplexity: `null`
- FP16: `true`

Fixed training writes:

- `outputs/logs/generator_fixed_full_train.log`
- `outputs/reports/generator_fixed_full_train_metrics.json`
- `outputs/reports/generator_fixed_train_diagnostics.json`

## Suspected Root Cause

Full-data inspection and batch diagnostics show the dataset is mostly usable and labels are not all ignored. The likely cause of the earlier `0.0` train loss / `NaN` eval loss is training instability from FP16 with this local FLAN-T5 checkpoint and insufficient explicit NaN/gradient diagnostics.

The fixed trainer disables FP16, explicitly masks only padding labels as `-100`, checks all-ignored label rates, clips gradients, and stops with diagnostic batches if non-finite loss or gradient norms occur.
