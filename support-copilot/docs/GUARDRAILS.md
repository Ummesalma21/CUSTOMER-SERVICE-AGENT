# Answerability And Safety Guardrails

The final system uses general guardrails to decide whether to answer, escalate, or reject. These rules are not tied to demo strings.

## Signals Used

- support keyword score from support-domain and support-action terms
- nearest KB similarity
- domain centroid similarity
- lexical/domain gate output
- account-specific patterns such as case, claim, account, transaction, or portal references
- evidence quality and citation availability
- answer-quality checks for fragments, missing citations, vague queries, and invalid formatting

## Policy

- `ANSWER`: used when retrieved evidence is strong, cited, and the generated or synthesized answer passes quality checks.
- `TICKET`: used for in-domain, support-like, account-specific, manual-review, or insufficient-evidence cases.
- `REJECT`: used conservatively when support keyword score is low and both KB/domain proximity signals are weak.

## Why Guardrails Exist

Plain RAG always answers, even when the KB has insufficient evidence. The guardrails reduce unsupported answers by routing uncertain support cases to tickets and clear out-of-domain cases to rejection.

## Limitations

The guardrails are heuristic and threshold-based. They are intended to make behavior safer and more explainable for this course project, not to replace production policy logic or human review.
