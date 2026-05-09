# Final Submission Checklist

- [x] README exists and includes setup, demo, training, and evaluation instructions.
- [ ] Conference-format report PDF is exported. Current repo has `report/report_outline.md`; export the final PDF separately.
- [ ] Slides `.pptx` or `.pdf` are exported. Add them under `slides/` or submit separately.
- [x] Code runs with CLI demo when local checkpoints/indexes are available.
- [x] Streamlit UI exists at `app_streamlit.py`.
- [x] Tool schema JSON exists at `schemas/tool_schema.json`.
- [x] Final metrics files exist under `outputs/reports/`.
- [x] Log directory exists at `outputs/logs/`; large logs are ignored by git.
- [x] Older intermediate results are archived under the top-level `archive/` folder.
- [x] Limitations are stated honestly in `README.md` and `outputs/reports/FINAL_RESULTS_FOR_REPORT.md`.
- [x] Exact demo-query strings are not used in inference decision logic. They appear only in docs, demo examples, regression checks, synthetic evaluation fixtures, or data fixtures.
- [x] `.gitignore` excludes local environments, caches, model binaries, indexes, raw data, large prediction dumps, and logs.

## Warnings Before Zipping

- Large checkpoints and FAISS indexes are ignored by git. Include them manually only if the course submission permits large artifacts, or rely on the regeneration commands in `SUBMISSION_NOTES.md`.
- The final report PDF and slide deck are not currently present as finished `.pdf`/`.pptx` files.
- FLAN-T5 is attempted only from local cache. If unavailable, the generator path falls back to extractive synthesis and should be described that way.
