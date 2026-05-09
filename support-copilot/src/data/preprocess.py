from __future__ import annotations

import random
import re
from pathlib import Path

from src.data.build_kb import chunk_documents
from src.data.load_multidoc2dial import load_multidoc2dial, offline_fixture
from src.data.make_negatives import reject_examples, ticket_examples
from src.data.make_preference_pairs import make_preference_pairs
from src.data.make_generator_data import make_generator_examples, split_generator_examples
from src.routing.domain_keywords import build_domain_keywords
from src.routing.centroids import build_domain_centroids
from src.retrieval.search_kb import embed_text, search
from src.utils.io import project_path, write_json, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger(__name__)


def prepare_data(config: dict) -> dict[str, int]:
    set_seed(int(config.get("seed", 42)))
    max_dialogues = config.get("max_dialogues") or config.get("max_triage_train_examples") or config.get("max_train_samples")
    data = load_multidoc2dial(
        bool(config.get("use_hf_dataset", True)),
        max_dialogues=None if config.get("data", {}).get("use_all_kb_docs") and not config.get("max_dialogues") else max_dialogues,
        max_documents=config.get("max_documents"),
        cache_dir=str(project_path("data", "hf_cache")) if config.get("use_hf_dataset", True) else None,
        require_hf=bool(config.get("require_hf_dataset", False)),
    )
    if config.get("include_fixture_docs") and data.get("source") != "offline_fixture":
        fixture = offline_fixture()
        data["documents"] = fixture["documents"] + data["documents"]
        data["dialogues"] = fixture["dialogues"] + data["dialogues"]
    chunking = config.get("chunking") if isinstance(config.get("chunking"), dict) else {}
    chunks = chunk_documents(
        data["documents"],
        max_words=int(chunking.get("max_words", config.get("chunk_max_words", 90))),
        min_words=int(chunking.get("min_words", config.get("chunk_min_words", 18))),
        sentence_overlap=int(chunking.get("sentence_overlap", config.get("chunk_sentence_overlap", 1))),
    )
    max_chunks = config.get("max_kb_chunks")
    if max_chunks:
        chunks = _balanced_chunk_cap(chunks, int(max_chunks))
    domains = sorted({c["domain"] for c in chunks})
    turns = _dialogue_turns(data["dialogues"], chunks)
    turns.extend(ticket_examples(domains, start=len(turns)))
    turns.extend(reject_examples(start=len(turns)))
    random.shuffle(turns)
    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")
    train_limit = int(config.get("max_triage_train_examples") or max_train or len(turns))
    eval_limit = int(max_eval) if max_eval else len(turns)
    train_rows = _balanced_rows(turns, train_limit, config.get("class_balance"))
    eval_rows = _balanced_rows(list(reversed(turns)), eval_limit, config.get("class_balance"))

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    retriever_train = [
        {"query": r["query"], "positive_chunk_id": r["gold_chunk_id"], "positive_doc_id": r["gold_doc_id"]}
        for r in train_rows
        if r.get("gold_triage") == "ANSWER" and r.get("gold_chunk_id")
    ]
    max_ret = config.get("max_retriever_train_pairs")
    if max_ret:
        retriever_train = retriever_train[: int(max_ret)]
    reranker_train = _make_reranker_rows(retriever_train, chunks, int(config.get("max_reranker_train_pairs") or 0))
    triage_train = train_rows
    pref_pairs = make_preference_pairs(eval_rows, chunks_by_id)
    max_pref = config.get("max_preference_pairs")
    if max_pref:
        pref_pairs = pref_pairs[: int(max_pref)]
    generator_examples = make_generator_examples(
        data["dialogues"],
        chunks,
        int(config.get("max_generator_train_examples") or 0) or None,
    )
    generator_train, generator_val, generator_test = split_generator_examples(generator_examples)
    keywords = build_domain_keywords(chunks)
    centroids = build_domain_centroids(chunks, keywords)

    out = project_path("data", "processed")
    write_jsonl(out / "kb_chunks.jsonl", chunks)
    write_jsonl(out / "dialogue_turns.jsonl", turns)
    write_jsonl(out / "retriever_train.jsonl", retriever_train)
    write_jsonl(out / "reranker_train.jsonl", reranker_train)
    write_jsonl(out / "triage_train.jsonl", triage_train)
    write_jsonl(out / "preference_pairs.jsonl", pref_pairs)
    write_jsonl(out / "generator_train.jsonl", generator_train)
    write_jsonl(out / "generator_val.jsonl", generator_val)
    write_jsonl(out / "generator_test.jsonl", generator_test)
    write_jsonl(out / "eval_set.jsonl", eval_rows)
    write_json(out / "domain_centroids.json", centroids)
    write_json(out / "domain_keywords.json", keywords)
    stats = {
        "kb_chunks": len(chunks),
        "dialogue_turns": len(turns),
        "retriever_train": len(retriever_train),
        "reranker_train": len(reranker_train),
        "triage_train": len(triage_train),
        "preference_pairs": len(pref_pairs),
        "generator_train": len(generator_train),
        "generator_val": len(generator_val),
        "generator_test": len(generator_test),
        "eval_set": len(eval_rows),
        "dataset_source": data.get("source", "unknown"),
    }
    write_json(project_path("outputs", "reports", "data_stats.json"), stats)
    LOG.info("Prepared data: %s", stats)
    return stats


def _dialogue_turns(dialogues: list[dict], chunks: list[dict]) -> list[dict]:
    rows: list[dict] = []
    indexes = _domain_indexes(chunks)
    for i, dialogue in enumerate(dialogues):
        domain = dialogue.get("domain", "unknown")
        for turn in dialogue.get("turns", []):
            text = turn.get("text") or turn.get("utterance") or ""
            speaker = str(turn.get("speaker", "unknown")).lower()
            if not text or not _is_user_speaker(speaker):
                continue
            best = None
            if domain in indexes:
                hits = search(text, top_k=1, index=indexes[domain])
                best = hits[0] if hits else None
            if not best:
                continue
            rows.append(
                {
                    "query_id": f"q{i:06d}_{len(rows):04d}",
                    "query": text,
                    "history": "",
                    "gold_doc_id": best["doc_id"],
                    "gold_chunk_id": best["chunk_id"],
                    "gold_domain": best["domain"],
                    "gold_triage": "ANSWER",
                    "gold_answer": _next_agent_text(dialogue.get("turns", []), turn) or _best_evidence_sentence(best["text"]),
                    "reference_answer": _next_agent_text(dialogue.get("turns", []), turn) or _best_evidence_sentence(best["text"]),
                }
            )
    return rows


def _is_user_speaker(speaker: str) -> bool:
    return speaker in {"user", "u", "customer", "client"} or "user" in speaker or "customer" in speaker


def _is_agent_speaker(speaker: str) -> bool:
    return speaker in {"agent", "assistant", "system", "wizard", "bot", "a"} or any(
        marker in speaker for marker in ["agent", "assistant", "system", "wizard"]
    )


def _next_agent_text(turns: list[dict], current_turn: dict) -> str | None:
    try:
        start = next(i for i, item in enumerate(turns) if item is current_turn)
    except StopIteration:
        return None
    for nxt in turns[start + 1 : start + 4]:
        if _is_user_speaker(str(nxt.get("speaker", "")).lower()):
            return None
        if _is_agent_speaker(str(nxt.get("speaker", "")).lower()):
            text = " ".join(str(nxt.get("text") or nxt.get("utterance") or "").split())
            if len(text.split()) >= 6:
                return text
    return None


def _best_evidence_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if len(sentence.split()) >= 6:
            return sentence.strip()
    return text.strip()


def _domain_indexes(chunks: list[dict]) -> dict[str, dict]:
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


def _balanced_chunk_cap(chunks: list[dict], limit: int) -> list[dict]:
    by_domain: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_domain.setdefault(chunk.get("domain", "unknown"), []).append(chunk)
    capped: list[dict] = []
    domains = sorted(by_domain)
    idx = 0
    while len(capped) < limit:
        added = False
        for domain in domains:
            if idx < len(by_domain[domain]):
                capped.append(by_domain[domain][idx])
                added = True
                if len(capped) >= limit:
                    break
        if not added:
            break
        idx += 1
    return capped


def _balanced_rows(rows: list[dict], limit: int, balance: dict | None) -> list[dict]:
    if not balance:
        return rows[:limit]
    buckets: dict[str, list[dict]] = {"ANSWER": [], "TICKET": [], "REJECT": []}
    for row in rows:
        buckets.setdefault(row.get("gold_triage", "ANSWER"), []).append(row)
    selected: list[dict] = []
    for label, frac in balance.items():
        key = str(label).upper()
        bucket = buckets.get(key, [])
        quota = int(limit * float(frac))
        if not bucket:
            continue
        for i in range(quota):
            row = dict(bucket[i % len(bucket)])
            if i >= len(bucket):
                row["query_id"] = f"{row.get('query_id', key.lower())}_dup{i:05d}"
            selected.append(row)
    if len(selected) < limit:
        seen = {r.get("query_id") for r in selected}
        for row in rows:
            if row.get("query_id") not in seen:
                selected.append(row)
                if len(selected) >= limit:
                    break
    return selected[:limit]


def _make_reranker_rows(retriever_rows: list[dict], chunks: list[dict], max_pairs: int = 0) -> list[dict]:
    rows: list[dict] = []
    for item in retriever_rows:
        pos = item["positive_chunk_id"]
        rows.append({"query": item["query"], "chunk_id": pos, "label": 1})
        pos_domain = next((c.get("domain") for c in chunks if c["chunk_id"] == pos), None)
        negs = [c["chunk_id"] for c in chunks if c["chunk_id"] != pos and c.get("domain") == pos_domain]
        negs.extend(c["chunk_id"] for c in chunks if c["chunk_id"] != pos and c.get("domain") != pos_domain)
        for neg in negs[:3]:
            rows.append({"query": item["query"], "chunk_id": neg, "label": 0})
            if max_pairs and len(rows) >= max_pairs:
                return rows[:max_pairs]
        if max_pairs and len(rows) >= max_pairs:
            return rows[:max_pairs]
    return rows
