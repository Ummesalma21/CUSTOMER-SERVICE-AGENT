# Baseline-0 vs Proposed With Reranker

Proposed config: `configs/final_eval_generator_fixed_reranker.yaml`
Reranker: `{'trained_pairs': 36000, 'positives': 9000, 'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'backend': 'cross-encoder', 'checkpoint': 'outputs\\reranker\\cross_encoder', 'device': 'cuda:0'}`

## Answer-Only

| Metric | Baseline-0 | Proposed + reranker | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.1040 | 0.1080 | +0.0040 |
| Recall@5 | 0.1820 | 0.2740 | +0.0920 |
| MRR@10 | 0.1291 | 0.1688 | +0.0396 |
| EvidenceHit@5 | 0.1820 | 0.2740 | +0.0920 |
| CitationPrecision | 1.0000 | 1.0000 | +0.0000 |
| GroundedAnswerRate | 1.0000 | 1.0000 | +0.0000 |
| UnsupportedClaimRate | 0.0000 | 0.0000 | +0.0000 |
| ESA | 0.4760 | 0.4720 | -0.0040 |
| AQS | 0.6270 | 0.5810 | -0.0460 |

## Unsupported-Answer Safety

| Metric | Baseline-0 | Proposed + reranker | Delta |
|---|---:|---:|---:|
| UnsupportedAnswerRate | 1.0000 | 0.5025 | -0.4975 |
| UnsupportedAnswerCount | 400 | 201 | -199.0000 |
| SafeActionRate | 0.0000 | 0.4975 | +0.4975 |
| OODAnswerRate | 1.0000 | 0.5100 | -0.4900 |
| TicketMissRate | 1.0000 | 0.4950 | -0.5050 |
| FalseRejectOnAnswerableRate | 0.0000 | 0.0100 | +0.0100 |
| UnsupportedAnswerPreventionCount | - | 199 | - |
| UnsupportedAnswerPreventionRate | - | 0.4975 | - |

## Mixed Workflow

| Metric | Baseline-0 | Proposed + reranker | Delta |
|---|---:|---:|---:|
| Tool Decision Accuracy | 0.6000 | 0.7150 | +0.1150 |
| ANSWER F1 | 0.7500 | 0.8116 | +0.0616 |
| TICKET F1 | 0.0000 | 0.5330 | +0.5330 |
| REJECT F1 | 0.0000 | 0.4908 | +0.4908 |
| Macro-F1 | 0.2500 | 0.6118 | +0.3618 |
| SupportedResponseRate | 0.5750 | 0.6930 | +0.1180 |
| UnsupportedAnswerRate | 0.4250 | 0.2760 | -0.1490 |
| EvidenceUseAccuracy | 0.6000 | 0.7150 | +0.1150 |
| OODAnswerRate | 1.0000 | 0.5100 | -0.4900 |
| TicketMissRate | 1.0000 | 0.4950 | -0.5050 |

## Efficiency / Latency

| Metric | Baseline-0 | Proposed + reranker | Delta |
|---|---:|---:|---:|
| avg_latency_ms | N/A | 90.5103 | - |
| p95_latency_ms | N/A | 269.5293 | - |
| throughput_qps | N/A | 11.0485 | - |
| avg_fraction_kb_searched | 1.0000 | 0.9112 | -0.0888 |
| global_fallback_rate | 1.0000 | 0.8660 | -0.1340 |
| avg_num_domains_searched | 0.0000 | 2.7810 | +2.7810 |
| avg_num_tool_calls | 1.0000 | 5.7320 | +4.7320 |
| REE@5 | 0.1820 | 0.3007 | +0.1187 |