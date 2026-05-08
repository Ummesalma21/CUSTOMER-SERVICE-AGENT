from __future__ import annotations


EASY_REJECTS = [
    "Who won the IPL match yesterday?",
    "Write Python code for merge sort.",
    "Give me a pasta recipe.",
    "What is the capital of France?",
    "Explain black holes.",
    "Tell me a joke about cats.",
]

HARD_REJECTS = [
    "How do I reset my Netflix password?",
    "Can I upgrade my iPhone warranty?",
    "Where is my Amazon package?",
    "Can I cancel my airline ticket?",
    "How do I claim health insurance reimbursement?",
    "Can I renew my passport here?",
]

NEAR_REJECTS = [
    "Can I update my bank KYC using this portal?",
    "How do I dispute my credit card transaction?",
    "Can I transfer college credits through this support page?",
    "How do I get a replacement SIM card?",
]

TICKET_PATTERNS = [
    "I followed the renewal instructions but my account still shows pending. Can someone check my case?",
    "The document says I can update my address, but the website gives an error. What should I do?",
    "My application was rejected but the KB does not explain the reason. Can support review it?",
]


def reject_examples(start: int = 0) -> list[dict]:
    rows = []
    for i, query in enumerate(EASY_REJECTS + HARD_REJECTS + NEAR_REJECTS, start=start):
        rows.append(
            {
                "query_id": f"reject_{i:04d}",
                "query": query,
                "history": "",
                "gold_doc_id": None,
                "gold_chunk_id": None,
                "gold_domain": None,
                "gold_triage": "REJECT",
            }
        )
    return rows


def ticket_examples(domains: list[str], start: int = 0) -> list[dict]:
    rows = []
    for i, query in enumerate(TICKET_PATTERNS, start=start):
        rows.append(
            {
                "query_id": f"ticket_{i:04d}",
                "query": query,
                "history": "",
                "gold_doc_id": None,
                "gold_chunk_id": None,
                "gold_domain": domains[i % len(domains)] if domains else "support",
                "gold_triage": "TICKET",
            }
        )
    return rows

