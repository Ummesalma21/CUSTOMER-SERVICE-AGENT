from __future__ import annotations


TOOLS = {
    "RouteDomain": "Predict likely KB domain using centroid similarity.",
    "SearchKB": "Retrieve relevant KB passages.",
    "GetPolicy": "Fetch full policy text for a known document or section.",
    "CreateTicket": "Create an escalation ticket.",
    "RejectQuery": "Reject out-of-domain queries.",
}


def trace(name: str, arguments: dict, returns: dict) -> dict:
    return {"name": name, "arguments": arguments, "returns": returns}

