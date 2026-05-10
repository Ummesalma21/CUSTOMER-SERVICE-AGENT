# Script Guide

Use these scripts for the final submission workflow.

## Demo / Evaluation

- `demo_cli.py`: one-query CLI demo.
- `evaluate_baseline_vs_proposed.py`: final Baseline vs Proposed comparison.
- `evaluate_esa_aqs.py`: ESA/AQS answer-quality proxy evaluation.
- `evaluate_unsupported_answer_safety.py`: unsupported-answer safety evaluation.
- `evaluate_mixed.py` and `evaluate_answer_only.py`: reusable evaluation helpers.

## Training / Reproduction

- `prepare_data.py`: prepare processed data.
- `build_index.py`: build the KB index.
- `train_retriever.py`: train the dense retriever.
- `train_reranker.py`: train the cross-encoder reranker.
- `train_triage.py`: train the DistilBERT triage/tool-policy model.
- `train_preference.py`: build/train the lightweight rubric preference module.
- `train_generator.py`: generator training.
- `build_balanced_triage_data.py`: balanced triage dataset builder.

Other scripts are development utilities retained for reproducibility/debugging, but they are not part of the main final demo/evaluation path.
