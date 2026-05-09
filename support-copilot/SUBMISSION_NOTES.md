# Submission Notes

## What To Include In The Code ZIP

Include the project source, configs, schema, README files, and final reports:

- `README.md`
- `requirements.txt`
- `run_all.py`
- `app_streamlit.py`
- `DEMO_UI.md`
- `SUBMISSION_NOTES.md`
- `FINAL_SUBMISSION_CHECKLIST.md`
- `configs/`
- `scripts/`
- `src/`
- `schemas/`
- `data/processed/domain_centroids.json`
- `data/processed/domain_keywords.json`
- `outputs/reports/`

Large local artifacts are ignored by git and should only be included in the ZIP if the submission system permits large files:

- `outputs/retriever/`
- `outputs/reranker/`
- `outputs/triage_balanced/`
- `outputs/preference/`
- `outputs/generator/`
- `data/indexes/`
- large prediction JSONL files under `outputs/reports/`

## If Checkpoints Are Missing

The code remains runnable for training and evaluation, but inference with the final configs expects local trained artifacts. Regenerate or restore them with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\train_retriever.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\build_index.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\train_reranker.py --config configs\full.yaml
.\.venv\Scripts\python.exe scripts\build_balanced_triage_data.py
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\triage_balanced.yaml
.\.venv\Scripts\python.exe scripts\train_generator.py --config configs\generator_finetune.yaml
```

## Final Demo Commands

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\final_eval_generator.yaml
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

## Final Evaluation Commands

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_mixed.py --config configs\final_eval_balanced_triage_best.yaml
.\.venv\Scripts\python.exe scripts\evaluate_grounding_mixed.py --config configs\final_eval_balanced_triage_best.yaml
.\.venv\Scripts\python.exe scripts\evaluate_answer_only.py --config configs\final_eval_generator.yaml
.\.venv\Scripts\python.exe scripts\evaluate_answer_quality.py --config configs\final_eval_generator.yaml
```

## Report And Slides

The repository currently contains `report/report_outline.md` plus generated result tables under `outputs/reports/`. A final conference-format PDF and slide deck should be exported separately before submission if required by the course portal.

## Artifact Policy

The git-tracked repository intentionally excludes virtual environments, caches, model binaries, FAISS indexes, raw data caches, and large prediction dumps. This keeps the code submission small and reproducible while preserving the final metric summaries.
