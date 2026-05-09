# Supported Synthesis Answer Improvement

Input: `C:\Users\ummes\OneDrive\Documents\New project\CUSTOMER-SERVICE-AGENT\support-copilot\outputs\reports\three_way_answer_only_predictions.jsonl`
Output: `C:\Users\ummes\OneDrive\Documents\New project\CUSTOMER-SERVICE-AGENT\support-copilot\outputs\reports\final_answer_only_supported_synthesis_predictions.jsonl`

This pass does not change retrieved hits, citations, routing, triage decisions, tickets, or rejects.
It only rewrites Proposed `ANSWER` text from the already selected top evidence passage using the extractive synthesizer.

Rows: `500`
Proposed ANSWER rows: `377`
ANSWER text rewritten: `99`

## ESA/AQS Result

| Metric | Baseline-0 Pretrained RAG | Proposed Supported Synthesis |
|---|---:|---:|
| ESA | 0.4760 | 0.6380 |
| AQS | 0.6270 | 0.7187 |

Decision and retrieval preservation check:

- decision changes: `0`
- top-5 hit changes: `0`
- answer text changes: `99`
