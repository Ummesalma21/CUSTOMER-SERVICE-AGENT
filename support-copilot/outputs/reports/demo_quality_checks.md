# Demo Quality Checks

Config: `configs/final_eval_generator.yaml`

## Benefits Renewal

Command:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Can I renew my benefits online?" --config configs\final_eval_generator.yaml
```

Observed:

- Decision: `ANSWER`
- Final answer: `You can renew eligible benefits online through the benefits portal.`
- Citation appears once in the citation block:
  - `doc_id=ssa_renewal_03`
  - `chunk_id=ssa_renewal_03_span0000`
  - `span=0-33`
- No duplicate inline citation.

## IPL

Command:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Who won the IPL yesterday?" --config configs\final_eval_generator.yaml
```

Observed:

- Decision: `REJECT`
- Tool: `RejectQuery`
- Reject reason: `out_of_domain`
- No citation shown.

## Account-Specific Benefits Issue

Command:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "My benefits renewal is stuck as pending for case ACCT-555123; can someone check my account?" --config configs\final_eval_generator.yaml
```

Observed:

- Decision: `TICKET`
- Tool: `CreateTicket`
- Category: `ssa`
- Severity: `medium`
- Ticket ID shown in the ticket block.

## Vague Query

Command:

```powershell
.\.venv\Scripts\python.exe scripts\demo_cli.py --query "Why am I here?" --config configs\final_eval_generator.yaml
```

Observed:

- Decision: `REJECT`
- Tool: `RejectQuery`
- Reject reason: `underspecified_or_out_of_scope`
- Final answer asks for a more specific support question.
- No random KB citation is shown.
