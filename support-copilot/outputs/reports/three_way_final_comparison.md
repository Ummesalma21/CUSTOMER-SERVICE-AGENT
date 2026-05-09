# Three-Way Final Comparison

## Systems

- Baseline-0: official simple RAG baseline using `sentence-transformers/all-MiniLM-L6-v2`, full KB search, no routing/reranker/triage/tools, always ANSWER.
- Baseline-1: fine-tuned retriever-only RAG ablation, full KB search, no routing/reranker/triage/tools, always ANSWER.
- Proposed: final support copilot with domain routing, triage/tool-policy, ticketing, rejection, and grounded answer validation.

Answer-only rows: `500`
Mixed workflow rows: `1000`

## Answer-Only Retrieval And Grounding

| System | Recall@1 | Recall@5 | MRR@10 | EvidenceHit@5 | CitationPrecision | GroundedAnswerRate | UnsupportedClaimRate | ESA | AQS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-0 Pretrained RAG | 0.1040 | 0.1820 | 0.1291 | 0.1820 | 1.0000 | 1.0000 | 0.0000 | 0.4760 | 0.6270 |
| Baseline-1 Fine-tuned RAG | 0.1580 | 0.3620 | 0.2327 | 0.3620 | 1.0000 | 1.0000 | 0.0000 | 0.7480 | 0.7840 |
| Proposed | 0.1560 | 0.3620 | 0.2320 | 0.3620 | 1.0000 | 1.0000 | 0.0000 | 0.5300 | 0.6733 |

## Mixed Workflow

| System | ToolAcc | ANSWER F1 | TICKET F1 | REJECT F1 | Macro-F1 | SupportedResponseRate | UnsupportedAnswerRate | EvidenceUseAccuracy | OODAnswerRate | TicketMissRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-0 Pretrained RAG | 0.6000 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5750 | 0.4250 | 0.6000 | 1.0000 | 1.0000 |
| Baseline-1 Fine-tuned RAG | 0.6000 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5750 | 0.4250 | 0.6000 | 1.0000 | 1.0000 |
| Proposed | 0.6760 | 0.7755 | 0.4541 | 0.5019 | 0.5772 | 0.6580 | 0.3190 | 0.6760 | 0.5500 | 0.5550 |

## Honest Reading

Baseline-0 is the official simple non-fine-tuned RAG assignment baseline. Baseline-1 is a stronger fine-tuned retriever-only ablation. If proposed improves over Baseline-0 but not over Baseline-1 on an answer-only metric, report that distinction rather than treating both baselines as the same system.