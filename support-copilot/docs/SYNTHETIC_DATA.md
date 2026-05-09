# Synthetic Ticket And Reject Data

## Why Synthetic Data Was Needed

MultiDoc2Dial mainly contains answerable document-grounded dialogue turns. The project also needs support workflow behavior: escalation/ticket creation and rejection of unsupported or out-of-domain requests. Synthetic TICKET and REJECT examples were added to evaluate those workflow decisions.

## Label Meanings

- `ANSWER`: the KB contains evidence that can support a direct cited response.
- `TICKET`: the query is in-domain and support-like, but requires account-specific/manual review or has insufficient KB evidence for a direct answer.
- `REJECT`: the query is out-of-domain, unsupported by the KB, or too vague to search responsibly.

## How Examples Were Generated

TICKET examples cover patterns such as:

- account or case status checks
- payment issues and duplicate charges
- document upload problems
- manual review requests
- portal/account-specific support issues

REJECT examples cover patterns such as:

- sports and current-events questions
- shopping or private-service support outside the KB
- unsupported public-service domains
- vague conversational queries

## Limitations

Synthetic workflow labels do not replace real production support logs. The mixed evaluation should be interpreted as a workflow-control test, not a claim of full real-world OOD robustness.
