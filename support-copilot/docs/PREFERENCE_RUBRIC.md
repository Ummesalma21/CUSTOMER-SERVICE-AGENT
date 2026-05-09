# Preference/Rubric Module

The project includes a lightweight preference-inspired scoring/ranking component. It is not full preference optimization of an LLM.

## What It Does

The rubric ranker scores candidate outputs using simple support-workflow criteria:

- helpfulness
- concise response style
- citation use for direct answers
- appropriate ticket creation for account-specific issues
- appropriate rejection for unsupported or out-of-domain queries

## What It Is Used For

The component satisfies the project requirement for a preference or rubric-based alignment step. It is best reported as a lightweight rubric/ranker and an ablation, not as the main source of final system improvement.

## Limitations

The rubric is much simpler than human preference data or policy optimization. It can over-reward cited direct answers and under-represent the value of safe `TICKET` or `REJECT` decisions, so workflow metrics and unsupported-answer safety are more important for the final claim.
