# Baseline-0 vs Proposed Final Comparison

Baseline-0 is the official simple pretrained RAG baseline. Proposed is the final routed/tool-using support copilot.

## Table 1: Answer-Only Retrieval And Grounding

| Metric | Baseline-0 Pretrained RAG | Proposed | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.1040 | 0.1560 | +0.0520 |
| Recall@5 | 0.1820 | 0.3620 | +0.1800 |
| MRR@10 | 0.1291 | 0.2320 | +0.1028 |
| EvidenceHit@5 | 0.1820 | 0.3620 | +0.1800 |
| CitationPrecision | 1.0000 | 1.0000 | +0.0000 |
| GroundedAnswerRate | 1.0000 | 1.0000 | +0.0000 |
| UnsupportedClaimRate | 0.0000 | 0.0000 | +0.0000 |

## Table 2: ESA/AQS

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| ESA | 0.4760 | 0.5300 | +0.0540 |
| AQS | 0.6270 | 0.6733 | +0.0463 |

## Table 3: Unsupported-Answer Safety

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| UnsupportedAnswerRate | 1.0000 | 0.5525 | -0.4475 |
| UnsupportedAnswerCount | 400 | 221 | -179.0000 |
| SafeActionRate | 0.0000 | 0.4475 | +0.4475 |
| OODAnswerRate | 1.0000 | 0.5500 | -0.4500 |
| TicketMissRate | 1.0000 | 0.5550 | -0.4450 |
| FalseRejectOnAnswerableRate | 0.0000 | 0.0000 | +0.0000 |
| UnsupportedAnswerPreventionCount | - | 179 | - |
| UnsupportedAnswerPreventionRate | - | 0.4475 | - |

## Table 4: Mixed Workflow / Triage

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| Tool Decision Accuracy | 0.6000 | 0.6760 | +0.0760 |
| ANSWER Precision | 0.6000 | 0.7018 | +0.1018 |
| ANSWER Recall | 1.0000 | 0.8667 | -0.1333 |
| ANSWER F1 | 0.7500 | 0.7755 | +0.0255 |
| TICKET Precision | 0.0000 | 0.4635 | +0.4635 |
| TICKET Recall | 0.0000 | 0.4450 | +0.4450 |
| TICKET F1 | 0.0000 | 0.4541 | +0.4541 |
| REJECT Precision | 0.0000 | 1.0000 | +1.0000 |
| REJECT Recall | 0.0000 | 0.3350 | +0.3350 |
| REJECT F1 | 0.0000 | 0.5019 | +0.5019 |
| Macro-F1 | 0.2500 | 0.5772 | +0.3272 |
| FalseRejectRate | 0.0000 | 0.0000 | +0.0000 |
| FalseAcceptRate | 1.0000 | 0.6650 | -0.3350 |
| OODAnswerRate | 1.0000 | 0.5500 | -0.4500 |
| TicketMissRate | 1.0000 | 0.5550 | -0.4450 |
| SupportedResponseRate | 0.5750 | 0.6580 | +0.0830 |
| UnsupportedAnswerRate | 0.4250 | 0.3190 | -0.1060 |
| EvidenceUseAccuracy | 0.6000 | 0.6760 | +0.0760 |

## Table 5: Efficiency / Latency

| Metric | Baseline-0 | Proposed | Delta |
|---|---:|---:|---:|
| avg_latency_ms | N/A | 42.2952 | - |
| p50_latency_ms | N/A | 43.2367 | - |
| p95_latency_ms | N/A | 56.8940 | - |
| throughput_qps | N/A | 23.6433 | - |
| total_eval_time_sec | N/A | 8.4590 | - |
| avg_fraction_kb_searched | 1.0000 | 0.9172 | -0.0828 |
| median_fraction_kb_searched | 1.0000 | 1.0000 | +0.0000 |
| p95_fraction_kb_searched | 1.0000 | 1.0000 | +0.0000 |
| global_fallback_rate | 1.0000 | 0.8720 | -0.1280 |
| avg_num_domains_searched | 0.0000 | 2.7990 | +2.7990 |
| avg_num_tool_calls | 1.0000 | 5.7730 | +4.7730 |
| REE@5 | 0.1820 | 0.3947 | +0.2127 |

## Table 6: Tool Usage

| Metric | Baseline-0 | Proposed |
|---|---:|---:|
| RouteDomain call rate | 0.0000 | 1.0000 |
| SearchKB call rate | 1.0000 | 0.9330 |
| GetPolicy call rate | 0.0000 | 0.8430 |
| CreateTicket call rate | 0.0000 | 0.1920 |
| RejectQuery call rate | 0.0000 | 0.0670 |
| average tool calls per query | 1.0000 | 5.7730 |

## Interpretation

- Proposed improves over the official baseline on answer-only retrieval: Recall@5 and EvidenceHit@5 improve from 0.1820 to 0.3620.
- Proposed improves ESA and AQS over the official baseline: ESA improves from 0.4760 to 0.5300; AQS improves from 0.6270 to 0.6733.
- Unsupported-answer safety is the fairer comparison for ticket/reject behavior because Baseline-0 has no triage tools.
- Baseline-0 always answers unsupported TICKET/REJECT cases, so its UnsupportedAnswerRate is 1.0 over unsupported cases.
- Proposed prevents a portion of these unsupported answers by using CreateTicket or RejectQuery.
- Latency should be read cautiously: Baseline-0 latency here is not live per-query inference timing, while proposed latency is loaded from saved traces when available.
- Proposed searches a smaller fraction of the KB on average through routing, but it uses more decision/tool logic.