# Final Results For Report

## Dataset Sizes

| Item | Count |
|---|---:|
| KB chunks | 5289 |
| MultiDoc2Dial dialogue turns | 57222 |
| Retriever train pairs | 9000 |
| Reranker train pairs | 36000 |
| Original triage train examples | 18000 |
| Balanced triage train examples | 18000: ANSWER 8000, TICKET 5000, REJECT 5000 |
| Balanced triage validation examples | 2000: ANSWER 1000, TICKET 500, REJECT 500 |
| Balanced triage test examples | 2000: ANSWER 1000, TICKET 500, REJECT 500 |
| Mixed workflow eval size | 1000: ANSWER 600, TICKET 200, REJECT 200 |
| Answer-only eval size | 1500 |

The balanced triage builder excluded exact query overlap from `data/processed/eval_mixed_1000.jsonl`. It found 896 unique mixed-eval queries and excluded 5878 overlapping answer-source rows; final train overlap with mixed eval was 0.

## Models

| Component | Model | Checkpoint |
|---|---|---|
| Retriever | `sentence-transformers/all-MiniLM-L6-v2` | `outputs/retriever/sentence_transformer` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `outputs/reranker/cross_encoder` |
| Old triage | `distilbert-base-uncased` | `outputs/triage/distilbert` |
| Balanced triage | `distilbert-base-uncased`, initialized from the local old triage checkpoint because network access was blocked | `outputs/triage_balanced/distilbert` |
| Preference | Lightweight rubric ranker | `outputs/preference` |

No retriever, reranker, preference ranker, KB index, or domain-centroid retraining was run for the balanced-triage stage.

## Balanced Triage Training

Config: `configs/triage_balanced.yaml`

Loss: cross entropy plus boundary margin, `cross_entropy + lambda_boundary * softplus(max_wrong - correct + mu)`.

Class weights: ANSWER `1.0`, TICKET `1.3`, REJECT `1.5`.

Validation metrics after epoch 3:

| Metric | Value |
|---|---:|
| Accuracy | 0.9985 |
| Macro-F1 | 0.9983 |
| ANSWER F1 | 0.9990 |
| TICKET F1 | 0.9990 |
| REJECT F1 | 0.9970 |
| TBP@0.15 | 0.9985 |

Validation confusion matrix:

| Gold \ Pred | ANSWER | TICKET | REJECT |
|---|---:|---:|---:|
| ANSWER | 998 | 0 | 2 |
| TICKET | 0 | 500 | 0 |
| REJECT | 0 | 1 | 499 |

Training log: `outputs/logs/triage_balanced_train.log`

Training metrics: `outputs/reports/triage_balanced_train_metrics.json`

## Final Mixed Workflow Evaluation

Config: `configs/final_eval_balanced_triage_best.yaml`

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| Tool Decision Accuracy | 0.600 | 0.765 |
| ANSWER Precision | 0.600 | 0.722 |
| ANSWER Recall | 1.000 | 0.988 |
| ANSWER F1 | 0.750 | 0.835 |
| TICKET Precision | 0.000 | 0.989 |
| TICKET Recall | 0.000 | 0.445 |
| TICKET F1 | 0.000 | 0.614 |
| REJECT Precision | 0.000 | 0.933 |
| REJECT Recall | 0.000 | 0.415 |
| REJECT F1 | 0.000 | 0.574 |
| Macro-F1 | 0.250 | 0.674 |
| FalseRejectRate | 0.000 | 0.0075 |
| FalseAcceptRate | 1.000 | 0.585 |
| OODAnswerRate | 1.000 | 0.585 |
| TicketMissRate | 1.000 | 0.555 |

Proposed confusion matrix:

| Gold \ Pred | ANSWER | TICKET | REJECT |
|---|---:|---:|---:|
| ANSWER | 593 | 1 | 6 |
| TICKET | 111 | 89 | 0 |
| REJECT | 117 | 0 | 83 |

The tuned system is intentionally conservative: it rejects only 6 of 600 answerable mixed-eval queries, while reject precision is 0.933. Reject recall and OODAnswerRate remain imperfect by design because answer correctness is prioritized over aggressive rejection.

## Mixed Workflow Grounding And Evidence Use

These metrics evaluate whether the system avoids unsupported direct answers when KB evidence is insufficient. For ANSWER rows, supported means the response cites correct or relevant KB evidence. For TICKET rows, supported means creating/escalating a ticket instead of giving a direct unsupported answer. For REJECT rows, supported means calling `RejectQuery`.

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| SupportedResponseRate | 0.551 | 0.718 |
| UnsupportedAnswerRate | 0.449 | 0.282 |
| EvidenceUseAccuracy | 0.600 | 0.765 |

This gives a grounding/evidence-use improvement over the answer-only baseline: the proposed workflow produces fewer unsupported direct answers on TICKET/REJECT cases while preserving near-baseline answer retrieval.

## Final Answer-Only Retrieval And Grounding

Config: `configs/final_eval_balanced_triage_best.yaml`

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| Recall@1 | 0.2567 | 0.2540 |
| Recall@5 | 0.4520 | 0.4513 |
| MRR@10 | 0.3296 | 0.3275 |
| EvidenceHit@5 | 0.4520 | 0.4513 |
| CitationPrecision | 1.000 | 1.000 |
| CitationRecall | 1.000 | 1.000 |
| GroundedAnswerRate | 1.000 | 1.000 |
| UnsupportedClaimRate | 0.000 | 0.000 |

The proposed system preserves near-baseline answer retrieval: Recall@5 is 0.4513 versus baseline 0.4520.

## Answer Quality Evaluation

Answer-quality metrics are computed only on answerable examples. Citation markers are stripped before token overlap and ROUGE-L scoring. The saved answer-only prediction file does not contain separate human reference answer text, so AnswerTokenF1 and ROUGE-L are not computed rather than inferred from evidence.

| Metric | Baseline RAG | Proposed Balanced Triage |
|---|---:|---:|
| AnswerTokenF1 | N/A | N/A |
| ROUGE-L | N/A | N/A |
| NoFragmentRate | 0.6967 | 0.6916 |
| CitationAttachedRate | 1.0000 | 1.0000 |
| DuplicateCitationRate | 1.0000 | 0.9447 |
| EmptyOrInvalidAnswerRate | 0.0907 | 0.0931 |
| AverageAnswerLengthWords | 19.7893 | 16.9683 |

The raw answer templates still produce some fragment-like outputs, so answer quality is not claimed to improve on fragment rate. Demo presentation now applies guardrails to avoid displaying fragment answers as confident responses and formats citations once in a separate block.

## Demo Interface

CLI:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\final_eval_balanced_triage_best.yaml
```

Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

If Streamlit is not installed:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

Example queries:

- `Can I renew my benefits online?`
- `Who won the IPL yesterday?`
- `My benefits renewal is stuck as pending for case ACCT-555123; can someone check my account?`
- `Why am I here?`

## Latency

Measured on the final mixed workflow evaluation:

| Metric | Value |
|---|---:|
| Average latency | 99.28 ms |
| p95 latency | 145.20 ms |
| Throughput | 10.07 qps |

## Demo Traces

Benefits renewal query:

- Query: `Can I renew my benefits online?`
- Decision: ANSWER
- Citation: `doc_id=ssa_renewal_03`, `chunk_id=ssa_renewal_03_span0000`
- Answer: `You can renew eligible benefits online through the benefits portal.`

Out-of-domain query:

- Query: `Who won the IPL yesterday?`
- Decision: REJECT
- Tool: `RejectQuery`

Account-specific in-domain issue:

- Query: `My benefits renewal is stuck as pending for case ACCT-555123; can someone check my account?`
- Decision: TICKET
- Tool: `CreateTicket`

## Limitations

TICKET and REJECT examples are partly synthetic because MultiDoc2Dial mainly contains answerable dialogue turns.

Reject is intentionally conservative to avoid rejecting answerable customer questions. This raises OODAnswerRate compared with a more aggressive reject policy, but it keeps FalseRejectRate low and Reject Precision high.

The main objective is not to beat baseline retrieval by a large amount. The desired result is to preserve answer quality while adding support workflow control. In the final answer-only evaluation, proposed Recall@5 is slightly below baseline by 0.0007, and this should be reported honestly.

The balanced triage model was initialized from the existing local DistilBERT triage checkpoint because direct Hugging Face access for `distilbert-base-uncased` was blocked in this environment.
