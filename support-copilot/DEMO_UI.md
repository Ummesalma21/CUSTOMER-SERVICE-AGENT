# Demo UI

Run the Streamlit demo from the `support-copilot` directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

If Streamlit is not installed:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

The UI uses `configs/final_eval_generator.yaml` by default when present, otherwise `configs/final_eval_balanced_triage_best.yaml`. It only runs one-query inference. It does not train, evaluate, or modify checkpoints.

The CLI demo remains available:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\final_eval_generator.yaml
```

Example queries:

- `Can I renew my benefits online?`
- `Who won the IPL yesterday?`
- `My benefits renewal is stuck as pending for case ACCT-555123; can someone check my account?`
- `Why am I here?`
