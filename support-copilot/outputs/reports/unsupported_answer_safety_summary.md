# Unsupported Answer Safety

Since Baseline-0 is a simple RAG system without ticket or reject tools, direct triage-F1 comparison can be misleading. We therefore report unsupported-answer safety: how often a system gives a direct answer when the KB does not contain sufficient evidence, and how many such cases the proposed system prevents.

Unsupported cases: `400`
Answerable cases: `600`

| Metric | Baseline-0 Pretrained RAG | Proposed | Delta |
|---|---:|---:|---:|
| UnsupportedAnswerRate | 1.0000 | 0.5525 | -0.4475 |
| UnsupportedAnswerCount | 400 | 221 | -179.0000 |
| SafeActionRate | 0.0000 | 0.4475 | +0.4475 |
| OODAnswerRate | 1.0000 | 0.5500 | -0.4500 |
| TicketMissRate | 1.0000 | 0.5550 | -0.4450 |
| FalseRejectOnAnswerableRate | 0.0000 | 0.0000 | +0.0000 |
| UnsupportedAnswerPreventionCount | - | 179 | - |
| UnsupportedAnswerPreventionRate | - | 0.4475 | - |
