# Reject-Aware Domain-Routed Customer Support Copilot

This project implements a reject-aware domain-routed customer-support copilot using MultiDoc2Dial. The system uses centroid-based domain routing, FAISS-compatible retrieval artifacts, cross-encoder-style reranking, a boundary-aware triage model for `ANSWER` / `TICKET` / `REJECT`, structured tool execution, and preference/rubric-based answer selection. It compares against a baseline retrieval-only RAG system and reports retrieval, grounding, triage, routing, and latency metrics.

The smoke path is intentionally lightweight: it uses deterministic hashing embeddings and a compact offline fixture. The `full_local` path downloads IBM/MultiDoc2Dial and trains real local models with small default limits so it can run on CPU-only machines, accepting that quality is not representative of a full training run.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Smoke Run

```bash
python run_all.py --config configs/smoke.yaml
```

This produces:

```text
outputs/reports/baseline_metrics.json
outputs/reports/proposed_metrics.json
outputs/reports/ablation_metrics.csv
outputs/reports/latency_metrics.json
outputs/reports/tool_traces.jsonl
outputs/reports/example_predictions.jsonl
outputs/reports/final_summary.md
```

## Full Local Run

```bash
python run_all.py --config configs/full_local.yaml
```

`full_local.yaml` is configured for small batches, one epoch per learned component, and no generator fine-tuning by default. It uses:

- Retriever: `sentence-transformers/all-MiniLM-L6-v2` fine-tuned with `MultipleNegativesRankingLoss`, then indexed as normalized embeddings plus `data/indexes/kb_index.faiss`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned on positive/negative query-passage pairs.
- Triage: `distilbert-base-uncased` sequence classifier trained with cross entropy plus `lambda_boundary * softplus(max_wrong_logit - correct_logit + mu)`.
- Preference: lightweight rubric ranker, intentionally not a neural model.

## Final GPU Experiment

Final config: `configs/full.yaml`.

The final run used GPU training after replacing CPU-only PyTorch with CUDA PyTorch:

- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- PyTorch: `2.11.0+cu128`.
- CUDA build: `12.8`.
- `torch.cuda.is_available()`: `True`.

The requested training limits were preserved. Evaluation was reduced from 3000 to 300 queries because the 3000-query evaluation exceeded the 1-hour command timeout; this reduction is recorded in `configs/full.yaml` and `outputs/reports/final_summary.md`.

Final dataset/training sizes:

- KB chunks: 5289.
- Dialogue turns: 57222.
- Retriever pairs: 9000.
- Reranker pairs: 36000.
- Triage examples: 18000.
- Preference pairs: 3000.
- Eval rows used in report: 300.

Final checkpoints:

- `outputs/retriever/sentence_transformer`
- `outputs/reranker/cross_encoder`
- `outputs/triage/distilbert`
- `outputs/preference`

Final metrics:

- Baseline: `Recall@5=0.4333`, `MRR@10=0.3134`, `EvidenceHit@5=0.4333`.
- Proposed: `Recall@5=0.1600`, `MRR@10=0.1259`, `Tool Decision Accuracy=0.4433`, `Macro-F1=0.2048`, `REE@5=0.6511`.
- Latency: average `1865.57ms`, p95 `4977.52ms`, `0.536 qps`.

The proposed system is worse than baseline on retrieval in this run. It does produce grounded cited answers when it answers (`CitationPrecision=1.0`, `CitationRecall=1.0`, `UnsupportedClaimRate=0.0`), but routing/triage calibration needs more work.

Regression/demo checks:

```bash
python scripts/regression_checks.py --config configs/full.yaml
python scripts/demo_cli.py --query "Can I renew my benefits online?" --config configs/full.yaml
python scripts/demo_cli.py --query "Who won the IPL yesterday?" --config configs/full.yaml
```

The benefits query now cites `ssa_renewal_03`; the IPL query calls `RejectQuery`.

## Calibrated No-Retrain Evaluation

Calibrated eval config: `configs/final_eval_calibrated.yaml`.

This run reused the saved full-training checkpoints and did not retrain. It changed only inference/evaluation behavior:

- Domain routing searches top-3 domains.
- Routed search falls back to global search when the routed best score is below `0.75`.
- REJECT is conservative: lexical gate low, centroid similarity low, and nearest-KB similarity below `0.40` must all hold.
- Uncertain support-like model rejects are softened to TICKET.
- Neural reranking is bypassed for this calibrated eval (`use_reranker: false`) because the 1000-query pass was otherwise dominated by cross-encoder/query-inference latency; retriever score ordering is used.
- The JSON index and query embeddings are cached in memory during evaluation.

Calibrated metrics on 1000 eval rows:

- Baseline: `Recall@5=0.437`, `MRR@10=0.3231`, `EvidenceHit@5=0.437`.
- Proposed: `Recall@5=0.423`, `MRR@10=0.3113`, `EvidenceHit@5=0.423`, `Tool Decision Accuracy=0.877`, `ANSWER F1=0.9345`, `TICKET F1=0.0`, `REJECT F1=0.0`, `False Reject Rate=0.122`, `False Accept Rate=0.0`, `REE@5=0.5656`.
- Latency: average `36.72ms`, p95 `47.09ms`, `27.23 qps`.

The calibrated retrieval metrics are much closer to baseline than the previous final run (`Recall@5=0.423` vs baseline `0.437`), but still slightly worse. Aggregate TICKET/REJECT F1 remain `0.0`; the operational regression checks still pass: benefits renewal cites `ssa_renewal_03`, and IPL calls `RejectQuery`.

## Step-by-Step

```bash
python scripts/prepare_data.py --config configs/full_local.yaml
python scripts/train_retriever.py --config configs/full_local.yaml
python scripts/build_index.py --config configs/full_local.yaml
python scripts/train_reranker.py --config configs/full_local.yaml
python scripts/train_triage.py --config configs/full_local.yaml
python scripts/train_preference.py --config configs/full_local.yaml
python scripts/evaluate.py --config configs/full_local.yaml
python scripts/demo_cli.py --config configs/full_local.yaml
```

## Acceptance Smoke Commands

```bash
python run_all.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml
python scripts/demo_cli.py --query "Can I renew my benefits online?" --config configs/smoke.yaml
python scripts/demo_cli.py --query "Who won the IPL yesterday?" --config configs/smoke.yaml
```

## Tools

The end-to-end system emits JSON traces for:

- `RouteDomain`
- `SearchKB`
- `GetPolicy`
- `CreateTicket`
- `RejectQuery`

Answers from KB evidence include `doc_id`, `chunk_id`, and span references. Out-of-domain questions call `RejectQuery`; related but under-supported questions call `CreateTicket`.

## Model Components

- Retriever: smoke-friendly hashing encoder for `smoke`; fine-tuned SentenceTransformer for `full_local`, saved under `outputs/retriever/sentence_transformer/`.
- Reranker: lexical scorer for `smoke`; fine-tuned CrossEncoder for `full_local`, saved under `outputs/reranker/cross_encoder/`.
- Triage/tool policy: boundary-aware lexical scorer for `smoke`; fine-tuned DistilBERT classifier for `full_local`, saved under `outputs/triage/distilbert/`.
- Preference alignment: rubric ranker saved under `outputs/preference/`.
- Generator: template-guided cited answer generation; FLAN-T5 LoRA hook is present but optional.

## Latest Small Full-Local Results

Run date: 2026-05-07 on CPU-only Torch (`torch.cuda.is_available() == False`).

Config limits: 300 balanced KB chunks, 64 train rows, 24 eval rows, one epoch for retriever/reranker/triage.

Stage results:

- Data: IBM/MultiDoc2Dial loaded; 904 dialogue turns, 63 retriever positives, 126 reranker pairs, 64 triage examples, 24 preference/eval rows.
- Retriever: trained 63 pairs, `train_loss=1.33`, `train_runtime=7.468s`.
- Reranker: trained 126 pairs, `train_loss=2.048`, `train_runtime=17.25s`.
- Triage: trained 64 examples, train accuracy `0.984375`, `TBP@0.15=0.984375`, last train loss `0.3565655`.
- Preference: trained 24 pairs, pair accuracy `1.0`.
- End-to-end proposed metrics on 24 eval rows: `Recall@5=0.1667`, `MRR@10=0.1458`, `Tool Decision Accuracy=0.375`, `Macro-F1=0.1818`, `False Reject Rate=0.3333`, `REE@5=0.6667`, average latency `227.09ms`.

These are smoke-sized CPU sanity results, not a claim of production quality. The baseline retrieval-only run scored `Recall@5=0.2083`, so the tiny full pipeline currently validates execution more than model quality.

Disk footprint after cleanup:

- `.venv`: about 1.2 GB.
- local Python `.python`: about 127 MB.
- project Hugging Face data cache: about 140 MB.
- saved checkpoints: retriever about 87 MB, reranker about 87 MB, triage about 256 MB.
- index artifacts: about 4 MB.

The initial Windows install logs and installer were removed. Duplicate user-level Hugging Face cache entries for the three downloaded models and IBM/MultiDoc2Dial were also removed after checkpoints were saved; unrelated user cache entries were left untouched.

## Notes

Every script accepts `--config`, writes predictable JSON/JSONL/CSV outputs, and falls back to CPU-compatible logic. The smoke mode does not require network access. `full_local` requires network on first run and uses `datasets>=2.18,<4` because IBM/MultiDoc2Dial is script-backed and newer `datasets` 4.x rejects dataset scripts.
