from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.answer_quality import CONTINUATION_WORDS, is_fragment_answer
from src.retrieval.search_kb import tokenize
from src.utils.io import project_path, read_jsonl, write_jsonl


BROKEN_PATTERNS = [
    "helpswith",
    "mostlong",
    "orskilled",
    "andmany",
    "seranyone",
    "benefitsand",
    "documentsand",
    "informationand",
]
VAGUE_QUERIES = {
    "yes",
    "no",
    "okay",
    "ok",
    "thanks",
    "thank you",
    "what about that",
    "tell me more",
    "can you help me",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/generator_train.jsonl")
    parser.add_argument("--val", default="data/processed/generator_val.jsonl")
    parser.add_argument("--test", default="data/processed/generator_test.jsonl")
    parser.add_argument("--tokenizer", default="google/flan-t5-small")
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()
    tokenizer = _load_tokenizer(args.tokenizer)
    split_paths = {"train": args.train, "val": args.val, "test": args.test}
    report = {"tokenizer": args.tokenizer, "splits": {}}
    samples = []
    rng = random.Random(42)
    for split, rel_path in split_paths.items():
        rows = read_jsonl(project_path(*Path(rel_path).parts))
        stats, worst = inspect_rows(rows, tokenizer)
        report["splits"][split] = stats
        for row in rng.sample(rows, min(args.sample_size, len(rows))):
            samples.append({"split": split, "sample_type": "random", **_sample_fields(row)})
        for item in worst[: args.sample_size]:
            samples.append({"split": split, "sample_type": "worst", **item})
    out_md = project_path("outputs", "reports", "generator_data_quality_report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(report), encoding="utf-8")
    write_jsonl(project_path("outputs", "reports", "generator_data_quality_samples.jsonl"), samples)
    print(json.dumps(report, indent=2))


def inspect_rows(rows: list[dict], tokenizer) -> tuple[dict, list[dict]]:
    if not rows:
        return {"rows": 0}, []
    lengths = {
        "query": [],
        "evidence": [],
        "target": [],
        "target_tokens": [],
    }
    counters = Counter()
    overlaps = {"evidence_target": [], "query_target": [], "evidence_query": []}
    pair_counts = Counter()
    worst = []
    for idx, row in enumerate(rows):
        query = str(row.get("query") or "")
        evidence_text = " ".join(str(e.get("text", "")) for e in row.get("evidence", []) if isinstance(e, dict))
        target = str(row.get("target") or row.get("target_answer") or row.get("reference_answer") or "")
        pair_counts[(query.strip().lower(), target.strip().lower())] += 1
        q_words = tokenize(query)
        e_words = tokenize(evidence_text)
        t_words = tokenize(target)
        lengths["query"].append(len(q_words))
        lengths["evidence"].append(len(e_words))
        lengths["target"].append(len(t_words))
        labels = tokenizer(text_target=target, truncation=True, max_length=96, padding="max_length")["input_ids"]
        non_pad = sum(1 for tok in labels if tok != tokenizer.pad_token_id)
        lengths["target_tokens"].append(non_pad)
        if not query.strip():
            counters["empty_query"] += 1
        if not evidence_text.strip():
            counters["empty_evidence"] += 1
        if not target.strip():
            counters["empty_target"] += 1
        if len(t_words) < 6:
            counters["target_lt_6_words"] += 1
        if len(t_words) < 8:
            counters["target_lt_8_words"] += 1
        if target.strip() and is_fragment_answer(target):
            counters["fragment_target"] += 1
        if target.strip()[:1] in string.punctuation:
            counters["target_starts_punctuation"] += 1
        if t_words and t_words[0].lower() in CONTINUATION_WORDS:
            counters["target_starts_continuation"] += 1
        if _is_vague_query(query):
            counters["vague_context_query"] += 1
        if _has_broken_spacing(query, evidence_text, target):
            counters["broken_spacing"] += 1
        if non_pad == 0:
            counters["all_ignored_if_pad_masked"] += 1
        ev_t = _overlap(e_words, t_words)
        q_t = _overlap(q_words, t_words)
        ev_q = _overlap(e_words, q_words)
        overlaps["evidence_target"].append(ev_t)
        overlaps["query_target"].append(q_t)
        overlaps["evidence_query"].append(ev_q)
        score = 0
        reasons = []
        for reason, condition in [
            ("empty_query", not query.strip()),
            ("empty_evidence", not evidence_text.strip()),
            ("empty_target", not target.strip()),
            ("short_target", len(t_words) < 8),
            ("fragment_target", bool(target.strip() and is_fragment_answer(target))),
            ("broken_spacing", _has_broken_spacing(query, evidence_text, target)),
            ("vague_query", _is_vague_query(query)),
            ("low_evidence_target_overlap", ev_t < 0.05),
        ]:
            if condition:
                score += 2 if reason.startswith("empty") else 1
                reasons.append(reason)
        if score:
            worst.append({"badness": score, "reasons": reasons, "row_index": idx, **_sample_fields(row), "evidence_target_overlap": ev_t})
    duplicate_pairs = sum(count - 1 for count in pair_counts.values() if count > 1)
    stats = {
        "rows": len(rows),
        **_rates(counters, len(rows)),
        "avg_query_words": _mean(lengths["query"]),
        "avg_evidence_words": _mean(lengths["evidence"]),
        "avg_target_words": _mean(lengths["target"]),
        "avg_target_tokens": _mean(lengths["target_tokens"]),
        "target_token_lengths": _distribution(lengths["target_tokens"]),
        "evidence_target_overlap_avg": _mean(overlaps["evidence_target"]),
        "query_target_overlap_avg": _mean(overlaps["query_target"]),
        "evidence_query_overlap_avg": _mean(overlaps["evidence_query"]),
        "duplicate_query_target_pairs": duplicate_pairs,
    }
    worst.sort(key=lambda item: item["badness"], reverse=True)
    return stats, worst


def _rates(counter: Counter, total: int) -> dict:
    out = {}
    for key, value in sorted(counter.items()):
        out[key] = value
        out[f"{key}_rate"] = value / max(1, total)
    return out


def _distribution(values: list[int]) -> dict:
    if not values:
        return {}
    return {
        "min": min(values),
        "p25": statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min(values),
        "median": statistics.median(values),
        "p75": statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max(values),
        "max": max(values),
    }


def _sample_fields(row: dict) -> dict:
    evidence = row.get("evidence", [])
    return {
        "example_id": row.get("example_id"),
        "query": row.get("query"),
        "target": row.get("target") or row.get("target_answer") or row.get("reference_answer"),
        "evidence": [e.get("text", "")[:500] for e in evidence[:3] if isinstance(e, dict)],
        "gold_doc_id": row.get("gold_doc_id"),
        "gold_chunk_id": row.get("gold_chunk_id"),
    }


def _is_vague_query(query: str) -> bool:
    q = re.sub(rf"[{re.escape(string.punctuation)}]", "", (query or "").lower()).strip()
    if q in VAGUE_QUERIES:
        return True
    words = q.split()
    return len(words) <= 3 and q in {"yes", "no", "okay", "thanks", "sure"}


def _has_broken_spacing(*texts: str) -> bool:
    joined = " ".join(t.lower() for t in texts)
    if any(pattern in joined for pattern in BROKEN_PATTERNS):
        return True
    return bool(re.search(r"\b[a-z]{10,}(and|with|or|for|to)[a-z]{4,}\b", joined))


def _overlap(left: list[str], right: list[str]) -> float:
    lset = {w for w in left if len(w) >= 4}
    rset = {w for w in right if len(w) >= 4}
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / max(1, min(len(lset), len(rset)))


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def _markdown(report: dict) -> str:
    lines = ["# Generator Data Quality Report", "", f"Tokenizer: `{report['tokenizer']}`", ""]
    for split, stats in report["splits"].items():
        lines.extend([f"## {split}", "", "| Metric | Value |", "|---|---:|"])
        for key, value in stats.items():
            if key == "target_token_lengths":
                continue
            lines.append(f"| {key} | {value} |")
        lines.extend(["", f"Target token lengths: `{stats.get('target_token_lengths', {})}`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
