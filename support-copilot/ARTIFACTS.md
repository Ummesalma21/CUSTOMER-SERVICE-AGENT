# Artifacts And Regeneration

## Included In The Code ZIP

The submission ZIP is intended to include:

- code: `src/`, `scripts/`, `run_all.py`, `app_streamlit.py`
- configs: `configs/`
- schemas: `schemas/tool_schema.json`
- documentation: `README.md`, `DEMO_UI.md`, `SUBMISSION_NOTES.md`, `ARTIFACTS.md`, `docs/`
- final report-facing outputs: `outputs/reports/`
- small processed metadata when present, such as domain keywords and centroids

## Not Included By Default

These artifacts are excluded or should be excluded because of size:

- `.venv/` and local Python installs
- Hugging Face caches
- raw downloaded datasets
- large processed training JSONL files
- FAISS/JSON indexes under `data/indexes/`
- trained checkpoint folders under `outputs/retriever/`, `outputs/reranker/`, `outputs/triage*/`, `outputs/generator/`
- large prediction dumps archived under `archive/old_outputs/`

Do not assume trained checkpoints are included unless the course ZIP explicitly contains them.

## Regenerate Missing Artifacts

Run from `support-copilot/` after installing requirements:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_retriever.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\build_index.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_reranker.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\build_balanced_triage_data.py
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\triage_balanced.yaml
.\.venv\Scripts\python.exe scripts\train_preference.py --config configs\train_full.yaml
.\.venv\Scripts\python.exe scripts\train_generator.py --config configs\generator_fixed.yaml
```

Expected regenerated locations:

- processed data: `data/processed/`
- indexes: `data/indexes/`
- retriever checkpoint: `outputs/retriever/`
- reranker checkpoint: `outputs/reranker/`
- triage checkpoints: `outputs/triage/`, `outputs/triage_balanced/`
- preference/rubric outputs: `outputs/preference/`
- generator checkpoint: `outputs/generator/`
- final reports: `outputs/reports/`

## Run Without Retraining

If checkpoints and indexes are already present:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\proposed_final.yaml
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
.\.venv\Scripts\python.exe scripts\evaluate_baseline0_vs_proposed.py
.\.venv\Scripts\python.exe scripts\evaluate_esa_aqs.py
.\.venv\Scripts\python.exe scripts\evaluate_unsupported_answer_safety.py
```

The final report-facing files are indexed in `outputs/reports/REPORT_INDEX.md`.
