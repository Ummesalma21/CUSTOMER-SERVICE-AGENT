# Generator And Answer Synthesis Summary

The final system uses generation only for customer-facing `ANSWER` wording. Tool decisions, citations, ticket creation, and rejection are handled by the routed support pipeline rather than by the generator.

## Role In The Final System

- Input: user query plus retrieved evidence passages.
- Output: concise answer text without generated citation strings.
- Citation handling: citations are attached by the system from retrieved KB evidence.
- Guardrails: answer text is validated for fragment control, citation deduplication, vague-query handling, and evidence sufficiency.
- Fallback: extractive supported synthesis is used when it gives a better evidence-supported answer formulation.

## Final Reported Answer-Quality Proxy Metrics

The headline ESA/AQS values are reported in:

- `outputs/reports/esa_aqs_metrics.json`
- `outputs/reports/esa_aqs_summary.md`
- `outputs/reports/baseline0_vs_proposed_supported_synthesis_metrics.json`

| Metric | Baseline-0 Pretrained RAG | Proposed |
|---|---:|---:|
| ESA | 0.4760 | 0.6380 |
| AQS | 0.6270 | 0.7187 |

ESA/AQS are automatic proxy metrics. They are useful for comparing evidence support and answer formulation under the same scoring rule, but they are not a substitute for human answer-quality evaluation.

## Submission Framing

The generator/synthesis component should be described as a bounded answer-formulation layer. The central project contribution remains the reject-aware support workflow: domain routing, cited KB retrieval, conservative rejection, ticket escalation, and unsupported-answer prevention.
