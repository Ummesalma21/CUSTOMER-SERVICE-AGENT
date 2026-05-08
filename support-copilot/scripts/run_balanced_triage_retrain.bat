@echo off
REM Run from repo root.

echo Building balanced triage dataset...
.\.venv\Scripts\python.exe scripts\build_balanced_triage_data.py
if errorlevel 1 exit /b 1

echo Training only balanced triage/tool-policy model...
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\triage_balanced.yaml
if errorlevel 1 exit /b 1

echo Done. Now point your eval config to outputs\triage_balanced\distilbert and run mixed evaluation.
