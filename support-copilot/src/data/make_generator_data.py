from __future__ import annotations

import re
from typing import Any

from src.generation.answer_quality import clean_answer_text, is_fragment_answer
from src.retrieval.search_kb import search


AGENT_ROLES = {"agent", "assistant", "system", "wizard", "bot", "a"}
USER_ROLES = {"user", "customer", "client", "u"}


def make_generator_examples(
    dialogues: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Create grounded generation examples from real user->agent turns.

    The preferred target is the next agent utterance after a user question. If
    the dataset row has evidence references, those are used first. Otherwise we
    retrieve the best in-domain passage and keep the example only when both the
    target answer and evidence look complete enough for supervised training.
    """
    indexes = _domain_indexes(chunks)
    by_chunk = {c["chunk_id"]: c for c in chunks}
    by_doc = _chunks_by_doc(chunks)
    examples: list[dict[str, Any]] = []
    for dialogue_idx, dialogue in enumerate(dialogues):
        domain = dialogue.get("domain", "unknown")
        turns = dialogue.get("turns", [])
        for turn_idx, turn in enumerate(turns):
            if not _is_user_turn(turn):
                continue
            query = clean_answer_text(turn.get("text") or turn.get("utterance") or "")
            if not _usable_query(query):
                continue
            answer = _next_agent_answer(turns, turn_idx)
            if not answer:
                continue
            evidence = _evidence_for_turn(turn, answer, domain, indexes.get(domain), by_chunk, by_doc)
            if not evidence:
                continue
            prompt = generator_prompt(query, evidence)
            examples.append(
                {
                    "example_id": f"gen_{dialogue_idx:06d}_{turn_idx:04d}",
                    "query": query,
                    "target_answer": answer,
                    "reference_answer": answer,
                    "gold_answer": answer,
                    "gold_domain": domain,
                    "gold_doc_id": evidence[0].get("doc_id"),
                    "gold_chunk_id": evidence[0].get("chunk_id"),
                    "evidence": evidence[:3],
                    "prompt": prompt,
                    "target": answer,
                    "source_type": "multidoc2dial_user_agent",
                }
            )
            if max_examples and len(examples) >= max_examples:
                return examples
    return examples


def generator_prompt(query: str, evidence: list[dict[str, Any]]) -> str:
    evidence_lines = [f"[{idx}] {item.get('text', '')}" for idx, item in enumerate(evidence[:3], start=1)]
    return (
        f"Question:\n{query}\n\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
        + "\n\nInstruction:\n"
        "Answer the question using only the evidence above. Write one concise customer-support answer in complete sentences. "
        "Do not include citations or document IDs in the answer text. If the evidence is insufficient, output exactly: INSUFFICIENT_EVIDENCE"
    )


def split_generator_examples(rows: list[dict[str, Any]], val_frac: float = 0.08, test_frac: float = 0.08) -> tuple[list, list, list]:
    n = len(rows)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    test = rows[:test_n]
    val = rows[test_n : test_n + val_n]
    train = rows[test_n + val_n :]
    return train, val, test


def _is_user_turn(turn: dict[str, Any]) -> bool:
    role = str(turn.get("speaker") or turn.get("role") or "").lower()
    return role in USER_ROLES or "user" in role or "customer" in role


def _is_agent_turn(turn: dict[str, Any]) -> bool:
    role = str(turn.get("speaker") or turn.get("role") or "").lower()
    return role in AGENT_ROLES or any(marker in role for marker in AGENT_ROLES)


def _next_agent_answer(turns: list[dict[str, Any]], start_idx: int) -> str | None:
    collected = []
    for turn in turns[start_idx + 1 : start_idx + 4]:
        if _is_user_turn(turn) and collected:
            break
        if not _is_agent_turn(turn):
            continue
        text = clean_answer_text(turn.get("text") or turn.get("utterance") or "")
        if _usable_answer(text):
            collected.append(text)
        if collected:
            break
    answer = " ".join(collected).strip()
    return answer if _usable_answer(answer) else None


def _evidence_for_turn(
    turn: dict[str, Any],
    answer: str,
    domain: str,
    index: dict | None,
    by_chunk: dict[str, dict],
    by_doc: dict[str, list[dict]],
) -> list[dict[str, Any]]:
    referenced = []
    for ref in _reference_ids(turn):
        if ref in by_chunk:
            referenced.append(by_chunk[ref])
        elif ref in by_doc and by_doc[ref]:
            referenced.append(by_doc[ref][0])
    if referenced:
        return referenced[:3]
    query = f"{turn.get('text', '')} {answer}"
    hits = search(query, top_k=3, index=index) if index else search(query, top_k=3, domain=domain)
    return [h for h in hits if _evidence_has_overlap(answer, h.get("text", ""))][:3] or hits[:1]


def _reference_ids(turn: dict[str, Any]) -> list[str]:
    raw = turn.get("raw") if isinstance(turn.get("raw"), dict) else turn
    refs = []
    for key in ["references", "reference", "doc_id", "doc_ids", "spans", "span_id", "span_ids"]:
        value = raw.get(key) if isinstance(raw, dict) else None
        if isinstance(value, str):
            refs.append(value)
        elif isinstance(value, list):
            refs.extend(str(v.get("doc_id") or v.get("id") or v) for v in value)
        elif isinstance(value, dict):
            refs.extend(str(v.get("doc_id") or v.get("id") or k) if isinstance(v, dict) else str(k) for k, v in value.items())
    return [r for r in refs if r]


def _domain_indexes(chunks: list[dict[str, Any]]) -> dict[str, dict]:
    from src.retrieval.search_kb import embed_text

    by_domain: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_domain.setdefault(chunk.get("domain", "unknown"), []).append(chunk)
    return {
        domain: {
            "dim": 128,
            "backend": "hashing-json",
            "chunks": [{**chunk, "embedding": embed_text(chunk["text"])} for chunk in domain_chunks],
        }
        for domain, domain_chunks in by_domain.items()
    }


def _chunks_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for chunk in chunks:
        out.setdefault(str(chunk.get("doc_id")), []).append(chunk)
    return out


def _usable_query(text: str) -> bool:
    words = re.findall(r"\b\w+\b", text)
    return len(words) >= 4 and text[-1:] in ".?!"


def _usable_answer(text: str) -> bool:
    if not text:
        return False
    if is_fragment_answer(text):
        return False
    words = re.findall(r"\b\w+\b", text)
    return len(words) >= 8


def _evidence_has_overlap(answer: str, evidence: str) -> bool:
    a = {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", answer)}
    e = {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", evidence)}
    if not a or not e:
        return False
    return len(a & e) / max(1, min(len(a), len(e))) >= 0.12
