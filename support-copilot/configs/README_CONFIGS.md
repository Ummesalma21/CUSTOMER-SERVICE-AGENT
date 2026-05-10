# Config Guide

Active configs are intentionally limited to final demo/evaluation and reproducible training.

## Final Evaluation And Demo

- `baseline.yaml`: official simple RAG baseline. Uses pretrained `sentence-transformers/all-MiniLM-L6-v2`, full-KB search, and always answers with a citation.
- `proposed.yaml`: main proposed support copilot. Uses the trained retriever/index, domain routing, balanced triage/tool-policy checkpoint, ticket/reject tools, the fine-tuned FLAN-T5-small generator when available, and grounded answer validation. Reranker is off in the headline setting.

## Training / Reproduction

- `train_full.yaml`: full data preparation and training pipeline for retriever, reranker, triage, preference/rubric module, and indexes.
- `train_triage_balanced.yaml`: balanced DistilBERT tool-policy/triage training.
- `train_generator.yaml`: FLAN-T5-small generator training config.

Older debug, smoke, threshold, and reranker-ablation configs are kept locally under `archive/old_configs/`, which is ignored by Git and excluded from the submission ZIP.
