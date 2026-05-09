# Two-Phase Answerability And Safety Policy

The final system uses a two-phase policy to decide whether to answer, escalate, or reject. This is intentional: customer-support systems should quickly filter obvious out-of-scope requests, while ambiguous support-like queries should be handled by learned and semantic evidence checks.

These rules are not tied to demo strings.

## Phase 1: Lexical Safety Gate

Phase 1 is a fast, interpretable guardrail. It handles only high-confidence cases:

- obvious out-of-domain categories such as sports, coding help, entertainment, recipes, shopping, and generic trivia
- vague or underspecified queries with weak support-domain signal
- account-specific or manual-review requests that should become tickets when KB evidence is insufficient

Phase 1 is deliberately conservative. It should not reject answerable support questions with strong KB evidence.

## Phase 2: Learned And Semantic Decision

Queries that are not resolved by Phase 1 continue to the main support pipeline:

- domain routing and domain centroid similarity
- nearest-KB similarity
- DistilBERT triage/tool-policy prediction
- KB evidence sufficiency checks
- citation availability
- answer-quality validation

Phase 2 chooses among `ANSWER`, `TICKET`, and `REJECT` using the learned triage model plus semantic retrieval signals.

## Signals Used

- support keyword score from support-domain and support-action terms
- nearest KB similarity
- domain centroid similarity
- lexical/domain gate output
- account-specific patterns such as case, claim, account, transaction, or portal references
- evidence quality and citation availability
- answer-quality checks for fragments, missing citations, vague queries, and invalid formatting
- DistilBERT triage/tool-policy probabilities

## Policy

- `ANSWER`: used when retrieved evidence is strong, cited, and the generated or synthesized answer passes quality checks.
- `TICKET`: used for in-domain, support-like, account-specific, manual-review, or insufficient-evidence cases.
- `REJECT`: used conservatively when support keyword score is low and both KB/domain proximity signals are weak.

## Why Guardrails Exist

Plain RAG always answers, even when the KB has insufficient evidence. The guardrails reduce unsupported answers by routing uncertain support cases to tickets and clear out-of-domain cases to rejection.

## Limitations

The Phase 1 guardrails are heuristic and threshold-based. They are intended to make behavior safer and more explainable for this course project, while Phase 2 provides the learned and semantic decision layer. Future work should learn more of the Phase 1 boundary from larger real support and out-of-domain logs.
