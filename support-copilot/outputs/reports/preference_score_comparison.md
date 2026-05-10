# Preference/Rubric Score Comparison

Rubric source: `src/preference/score_candidates.py`
Preference/rubric summary: `{'trained_pairs': 1000, 'pair_accuracy': 1.0, 'model': 'rubric-ranker'}`
Rubric artifact: `{'rubric': 'citation+grounding+tool+concise'}`

The score is a lightweight rubric score, not DPO/RLHF. It rewards cited answers, insufficient-evidence ticket style, out-of-domain rejection style, concise outputs, and an extra cited-answer bonus for ANSWER examples.

## Answer-Only

| System | Mean Preference Score | Win Rate vs Baseline | Rows |
|---|---:|---:|---:|
| Baseline | 4.9680 | - | 500 |
| Baseline-1 Fine-tuned RAG | 4.9760 | 0.0240 | 500 |
| Proposed | 4.1200 | 0.0200 | 500 |

## Mixed Workflow

| System | Mean Preference Score | Win Rate vs Baseline | Rows |
|---|---:|---:|---:|
| Baseline | 4.5810 | - | 1000 |
| Baseline-1 Fine-tuned RAG | 4.5690 | 0.0140 | 1000 |
| Proposed | 3.9990 | 0.0160 | 1000 |
