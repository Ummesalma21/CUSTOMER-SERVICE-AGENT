# Evaluation Metrics

This project evaluates three different capabilities: answer retrieval, support workflow control, and evidence-supported answer quality. The metrics below use the same definitions for Baseline-0 and the proposed system unless noted.

## Answer-Only Retrieval And Grounding

These metrics are computed only on answerable MultiDoc2Dial-derived examples.

- **Recall@1**: fraction of answerable queries where the gold evidence appears as the top retrieved passage.
- **Recall@5**: fraction of answerable queries where the gold evidence appears in the top 5 retrieved passages.
- **MRR@10**: mean reciprocal rank of the first correct evidence passage within the top 10.
- **EvidenceHit@5**: fraction of answerable queries where any top-5 passage matches the expected evidence. In this project it tracks Recall@5.
- **CitationPrecision**: fraction of emitted citations that point to a retrieved KB passage.
- **GroundedAnswerRate**: fraction of answer outputs that include a system-attached citation.
- **UnsupportedClaimRate**: fraction of answer outputs that contain a direct answer without an attached supporting citation.

## ESA And AQS

These are automatic proxy metrics for evidence support and answer formulation. They are not human evaluation.

- **ESA, Evidence Support Accuracy**: binary per answer. A row passes when a citation exists, cited evidence text is found, the citation is relevant to the query, the answer is semantically supported by the citation, the answer addresses the query, and the answer is not malformed.
- **AQS, Answer Quality Score**: 0-to-1 composite score: `(Fluency + Correctness + Trueness) / 6`. Each component is scored 0, 1, or 2.
- **Fluency**: checks whether the answer is complete, non-empty, non-fragmentary, and readable.
- **Correctness / Directness**: checks whether the answer directly addresses the user query.
- **Trueness / Grounding**: checks whether cited evidence supports the answer.

The final ESA/AQS run uses the same similarity thresholds for Baseline-0 and Proposed:

- query-citation similarity >= `0.35`
- answer-citation similarity >= `0.40`
- query-answer similarity >= `0.30`

## Mixed Workflow Metrics

These metrics are computed on `data/processed/eval_mixed_1000.jsonl`, which contains 600 `ANSWER`, 200 `TICKET`, and 200 `REJECT` examples.

- **Tool Decision Accuracy**: fraction of examples where the predicted decision matches `gold_decision`.
- **ANSWER Precision / Recall / F1**: class metrics for the `ANSWER` decision.
- **TICKET Precision / Recall / F1**: class metrics for the `TICKET` decision.
- **REJECT Precision / Recall / F1**: class metrics for the `REJECT` decision.
- **Macro-F1**: unweighted mean of ANSWER F1, TICKET F1, and REJECT F1.
- **FalseRejectRate**: fraction of answerable examples incorrectly predicted as `REJECT`.
- **FalseAcceptRate**: fraction of unsupported examples, gold `TICKET` or `REJECT`, incorrectly predicted as `ANSWER`.
- **OODAnswerRate**: fraction of gold `REJECT` examples predicted as `ANSWER`.
- **TicketMissRate**: fraction of gold `TICKET` examples predicted as `ANSWER`.

## Unsupported-Answer Safety

These metrics are used because Baseline-0 has no ticket/reject tools and always answers.

- **Unsupported case**: an example whose gold decision is `TICKET` or `REJECT`.
- **Unsupported answer**: a direct `ANSWER` on an unsupported case.
- **UnsupportedAnswerRate**: unsupported answers divided by unsupported cases.
- **UnsupportedAnswerCount**: number of unsupported cases directly answered.
- **UnsupportedAnswerPreventionCount**: number of Baseline-0 unsupported answers where Proposed chooses `TICKET` or `REJECT`.
- **UnsupportedAnswerPreventionRate**: prevention count divided by Baseline-0 unsupported answer count.
- **SafeActionRate**: fraction of unsupported cases predicted as `TICKET` or `REJECT`.
- **FalseRejectOnAnswerableRate**: fraction of answerable examples predicted as `REJECT`.

## Mixed Evidence-Use Metrics

- **SupportedResponseRate**: fraction of examples where the response type matches the evidence situation: cited `ANSWER` for answerable examples, `CreateTicket` for ticket examples, and `RejectQuery` for reject examples.
- **UnsupportedAnswerRate**: fraction of mixed examples where the system gives a direct answer when the gold label indicates ticketing or rejection.
- **EvidenceUseAccuracy**: fraction of examples where the system selects the correct evidence-use action: citation for `ANSWER`, ticket tool for `TICKET`, reject tool for `REJECT`.

## Efficiency Metrics

- **avg_latency_ms / p50_latency_ms / p95_latency_ms**: per-query latency statistics.
- **throughput_qps**: queries processed per second.
- **avg_fraction_kb_searched**: approximate fraction of KB chunks searched per query.
- **global_fallback_rate**: fraction of queries where routed search fell back to full-KB search.
- **avg_num_domains_searched**: average number of routed domains searched per query.
- **avg_num_tool_calls**: average number of tool calls per query.
- **REE@5**: retrieval efficiency estimate, `EvidenceHit@5 / avg_fraction_kb_searched`.

## Preference/Rubric Metric

- **Mean Preference Score**: score from a lightweight rubric ranker that rewards citation use, concise grounded answers, appropriate ticket language, and appropriate rejection language. It is not DPO, RLHF, or full preference optimization.
