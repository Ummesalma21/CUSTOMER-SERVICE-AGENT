# Balanced Triage Dataset Builder

This kit builds the balanced `ANSWER / TICKET / REJECT` triage dataset needed after threshold tuning hit its limit.

## Why this is needed

Your original triage model was too answer-biased because MultiDoc2Dial mostly provides answerable document-grounded turns. The mixed evaluation improved workflow behavior, but OOD answer rate remained high. That means the triage/tool-policy model needs a better balanced training set.

## Files to copy into your repo

```text
scripts/build_balanced_triage_data.py
configs/triage_balanced.yaml
scripts/run_balanced_triage_retrain.bat
```

## Build dataset

From repo root:

```bat
.\.venv\Scripts\python.exe scripts\build_balanced_triage_data.py
```

It creates:

```text
data/processed/triage_train_balanced.jsonl
data/processed/triage_val_balanced.jsonl
data/processed/triage_test_balanced.jsonl
outputs/reports/triage_balanced_dataset_summary.json
```

## Target sizes

```text
Train: ANSWER 8000, TICKET 5000, REJECT 5000
Val:   ANSWER 1000, TICKET 500,  REJECT 500
Test:  ANSWER 1000, TICKET 500,  REJECT 500
```

## Label definitions

```text
ANSWER = answerable from KB evidence
TICKET = in-domain but account-specific/manual-review issue
REJECT = out-of-domain query
```

## Important

The builder extracts ANSWER examples from your local `data/processed/**/*.jsonl` files, so run it inside the repo after MultiDoc2Dial preprocessing has completed.

By default, it excludes exact query overlap with `data/processed/eval_mixed_1000.jsonl` if that file exists.

## Retrain only triage

```bat
.\.venv\Scripts\python.exe scripts\train_triage.py --config configs\triage_balanced.yaml
```

Then create/update your final eval config to use:

```text
outputs/triage_balanced/distilbert
```

Do not retrain retriever or reranker.
