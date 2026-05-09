# Archive Notes

This top-level archive preserves old configs, reports, logs, outputs, and run snapshots that are useful for auditability but should not dominate the final submission narrative.

- `old_configs/`: smoke, debug, historical final-eval variants, generator experiments, and rechunking configs.
- `old_reports/`: small markdown/json summaries from intermediate or duplicate experiments.
- `old_outputs/`: bulky prediction dumps, scored JSONL files, and CSV sweeps. This folder is git-ignored to avoid pushing large artifacts.
- `old_logs/`: training and evaluation logs. Log files are git-ignored.
- `old_runs/`: older full run snapshots. This folder is git-ignored.

The active final results are in `outputs/reports/`.
