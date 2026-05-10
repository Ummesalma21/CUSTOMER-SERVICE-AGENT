# Submission Notes

## What To Zip

Include:

- source code: `src/`, `scripts/`, `schemas/`, `run_all.py`, `app_streamlit.py`
- docs: `README.md`, `DEMO_UI.md`, `SUBMISSION_NOTES.md`, `FINAL_SUBMISSION_CHECKLIST.md`, `docs/`
- artifact details: `ARTIFACTS.md`
- final configs: `configs/`
- small processed metadata needed by routing: `data/processed/domain_centroids.json`, `data/processed/domain_keywords.json`
- final report-facing outputs: `outputs/reports/`

Do not include local environments, caches, raw downloaded datasets, or huge checkpoint binaries unless the submission portal explicitly requires them.

Old experiments are kept locally under `archive/` and are excluded from the submitted ZIP/GitHub copy to keep the submission focused and reproducible.

## What Not To Zip By Default

- `.venv/`
- `.python/`
- `data/raw/`
- `data/cache/`
- `data/hf_cache/`
- `data/indexes/`
- `outputs/retriever/`
- `outputs/reranker/`
- `outputs/triage_balanced/`
- `outputs/generator/`
- large prediction dumps in `archive/old_outputs/`
- local `archive/` experiment backups

## Main Configs

- `configs/baseline.yaml`: official Baseline.
- `configs/proposed.yaml`: main proposed final system, reranker off.
- `configs/train_full.yaml`: full training/reproduction.
- `configs/train_triage_balanced.yaml`: balanced triage/tool-policy training.
- `configs/train_generator.yaml`: generator training.

## Main Commands

Demo:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\proposed.yaml
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

Official final comparison:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baseline_vs_proposed.py
.\.venv\Scripts\python.exe scripts\evaluate_esa_aqs.py
.\.venv\Scripts\python.exe scripts\evaluate_unsupported_answer_safety.py
```

Reproduction training:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_retriever.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\build_index.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_reranker.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\build_balanced_triage_data.py
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\train_triage_balanced.yaml
```

## Final Results

Final results are in:

- `outputs/reports/FINAL_RESULTS_FOR_REPORT.md`
- `outputs/reports/REPORT_INDEX.md`
- `outputs/reports/baseline_vs_proposed_summary.md`
- `outputs/reports/esa_aqs_summary.md`
- `outputs/reports/unsupported_answer_safety_summary.md`

Metric definitions are in `docs/METRICS.md`. The two-phase policy is described in `docs/GUARDRAILS.md`.

Older experiments are kept in the local ignored `archive/` folder and are not part of the submitted GitHub/ZIP copy.

## Checkpoints

Final inference expects local trained artifacts under `outputs/` and indexes under `data/indexes/`. These are intentionally git-ignored because they are large. If checkpoints are missing, regenerate them with the training commands above or restore them from the course artifact bundle.

See `ARTIFACTS.md` for the full included/excluded artifact list and regeneration path.
