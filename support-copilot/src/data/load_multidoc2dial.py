from __future__ import annotations
from typing import Any


def load_multidoc2dial(
    use_hf: bool = True,
    max_dialogues: int | None = None,
    max_documents: int | None = None,
    cache_dir: str | None = None,
    require_hf: bool = False,
) -> dict[str, Any]:
    """Load IBM/multidoc2dial or a compact offline fixture with the same shape."""
    if use_hf:
        try:
            from datasets import load_dataset  # type: ignore

            docs_ds = load_dataset("IBM/multidoc2dial", "document_domain", cache_dir=cache_dir, trust_remote_code=True)
            dial_ds = load_dataset("IBM/multidoc2dial", "dialogue_domain", cache_dir=cache_dir, trust_remote_code=True)
            normalized = _normalize_hf(docs_ds, dial_ds, max_dialogues, max_documents)
            if normalized["documents"] and normalized["dialogues"]:
                normalized["source"] = "IBM/multidoc2dial"
                return normalized
        except Exception as exc:
            if require_hf:
                raise RuntimeError(f"Could not load IBM/multidoc2dial: {exc}") from exc
    fixture = offline_fixture()
    fixture["source"] = "offline_fixture"
    return fixture


def _normalize_hf(docs_ds: Any, dial_ds: Any, max_dialogues: int | None, max_documents: int | None) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    for split in docs_ds:
        for row in docs_ds[split]:
            if max_documents and len(docs) >= max_documents:
                break
            if row.get("doc_data"):
                docs.extend(_docs_from_doc_data(row["doc_data"], max_documents, len(docs)))
                continue
            domain = row.get("domain") or row.get("doc_domain") or split
            doc_id = str(row.get("doc_id") or row.get("id") or f"{domain}_{len(docs)}")
            title = row.get("title") or row.get("doc_title") or doc_id
            text = row.get("doc_text") or row.get("text") or row.get("content") or ""
            if isinstance(text, dict):
                text = " ".join(str(v) for v in text.values())
            docs.append({"doc_id": doc_id, "domain": domain, "title": title, "text": str(text)})
        if max_documents and len(docs) >= max_documents:
            break
    dialogues: list[dict[str, Any]] = []
    for split in dial_ds:
        for row in dial_ds[split]:
            if max_dialogues and len(dialogues) >= max_dialogues:
                break
            if row.get("dial_data"):
                dialogues.extend(_dialogues_from_dial_data(row["dial_data"], max_dialogues, len(dialogues)))
                continue
            domain = row.get("domain") or row.get("doc_domain") or split
            turns = row.get("turns") or row.get("utterances") or []
            if isinstance(turns, dict):
                users = turns.get("user_utterance") or turns.get("utterance") or []
                turns = [{"speaker": "user", "text": t} for t in users]
            dialogues.append({"domain": domain, "turns": turns, "raw": row})
    return {"documents": docs, "dialogues": dialogues}


def _docs_from_doc_data(doc_data: dict[str, Any], max_documents: int | None = None, current: int = 0) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for domain, domain_docs in doc_data.items():
        if isinstance(domain_docs, list):
            iterable = enumerate(domain_docs)
        else:
            iterable = domain_docs.items()
        for key, doc in iterable:
            if max_documents and current + len(docs) >= max_documents:
                return docs
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("doc_id") or key)
            spans = doc.get("spans") or {}
            if spans:
                if isinstance(spans, dict):
                    text = " ".join(str(s.get("text", "")) for s in spans.values() if isinstance(s, dict))
                else:
                    text = " ".join(str(s.get("text", "")) for s in spans if isinstance(s, dict))
            else:
                text = str(doc.get("doc_text") or doc.get("text") or "")
            docs.append({"doc_id": doc_id, "domain": domain, "title": doc.get("title", doc_id), "text": text})
    return docs


def _dialogues_from_dial_data(dial_data: dict[str, Any], max_dialogues: int | None, current: int) -> list[dict[str, Any]]:
    dialogues: list[dict[str, Any]] = []
    for domain, domain_dialogues in dial_data.items():
        for dialogue in domain_dialogues:
            if max_dialogues and current + len(dialogues) >= max_dialogues:
                return dialogues
            turns = []
            for turn in dialogue.get("turns", []):
                text = turn.get("utterance") or turn.get("text") or ""
                role = turn.get("role") or turn.get("speaker") or ""
                turns.append({"speaker": role or "user", "text": text})
            dialogues.append({"domain": domain, "turns": turns, "raw": dialogue})
            continue
        if isinstance(domain_dialogues, dict):
            iterable = domain_dialogues.values()
        else:
            iterable = []
        for dialogue in iterable:
            if max_dialogues and current + len(dialogues) >= max_dialogues:
                return dialogues
            turns = []
            for turn in dialogue.get("turns", []):
                text = turn.get("utterance") or turn.get("text") or ""
                role = turn.get("role") or turn.get("speaker") or ""
                turns.append({"speaker": role or "user", "text": text})
            dialogues.append({"domain": domain, "turns": turns, "raw": dialogue})
    return dialogues


def offline_fixture() -> dict[str, Any]:
    documents = [
        {
            "doc_id": "ssa_renewal_03",
            "domain": "ssa",
            "title": "Benefits renewal",
            "text": (
                "You can renew eligible benefits online through the benefits portal. "
                "The renewal flow requires identity verification, current address, and income information. "
                "If the portal shows pending after submission, support must review the case."
            ),
        },
        {
            "doc_id": "ssa_address_02",
            "domain": "ssa",
            "title": "Address updates",
            "text": (
                "Benefit recipients can update their mailing address online or by phone. "
                "Changes may take several business days to appear. Account-specific errors require support review."
            ),
        },
        {
            "doc_id": "dmv_license_01",
            "domain": "dmv",
            "title": "Driver license renewal",
            "text": (
                "Drivers may renew a license online when the renewal notice says online renewal is allowed. "
                "A vision test or unpaid fee can require an office visit."
            ),
        },
        {
            "doc_id": "dmv_vehicle_04",
            "domain": "dmv",
            "title": "Vehicle registration",
            "text": (
                "Vehicle registration renewal can be completed online with plate number, insurance status, and payment. "
                "The receipt should be kept until the new registration arrives."
            ),
        },
        {
            "doc_id": "va_claims_01",
            "domain": "va",
            "title": "Veterans claims",
            "text": (
                "Veterans can check claim status in the online claim portal. "
                "Medical evidence and service records may be required before a benefits claim is decided."
            ),
        },
    ]
    dialogues = [
        {"domain": "ssa", "turns": [{"speaker": "user", "text": "Can I renew my benefits online?"}]},
        {"domain": "ssa", "turns": [{"speaker": "user", "text": "The renewal page still says pending. Can someone check my case?"}]},
        {"domain": "ssa", "turns": [{"speaker": "user", "text": "How do I update my mailing address for benefits?"}]},
        {"domain": "dmv", "turns": [{"speaker": "user", "text": "Can I renew my driver license online?"}]},
        {"domain": "dmv", "turns": [{"speaker": "user", "text": "What do I need to renew vehicle registration?"}]},
        {"domain": "va", "turns": [{"speaker": "user", "text": "How can I check my veterans claim status?"}]},
    ]
    return {"documents": documents, "dialogues": dialogues}
