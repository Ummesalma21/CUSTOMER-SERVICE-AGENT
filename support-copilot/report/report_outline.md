# Report Outline

1. Problem statement
2. Related work: ReAct, ToolLLM, rubric-based ranking, PEFT/LoRA, support RAG
3. Proposed two-phase reject-aware domain-routed RAG method
   - Phase 1: answerability guardrail for high-confidence OOD, vague, and account-specific cases
   - Phase 2: learned and semantic decision using routing, KB similarity, DistilBERT triage, and evidence validation
4. Tool schema table
5. Boundary-aware triage loss
6. Evaluation metric definitions: retrieval, workflow, unsupported-answer safety, ESA/AQS, and efficiency
7. MultiDoc2Dial preprocessing and synthetic REJECT/TICKET data
8. Baseline and proposed results
9. Ablations
10. Latency and throughput
11. Example tool traces
12. Limitations
13. Future work
