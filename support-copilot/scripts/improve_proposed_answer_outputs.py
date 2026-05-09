from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.extractive_synthesizer import synthesize_extractive_answer
from src.generation.templates import cited_answer
from src.utils.io import project_path, read_jsonl, write_jsonl
from scripts.evaluate_esa_aqs import _load_embedder, _score_answer


INPUT = project_path("outputs", "reports", "three_way_answer_only_predictions.jsonl")
OUTPUT = project_path("outputs", "reports", "final_answer_only_supported_synthesis_predictions.jsonl")
SUMMARY = project_path("outputs", "reports", "supported_synthesis_answer_improvement_summary.md")


def main() -> None:
    rows = read_jsonl(INPUT)
    embedder = _load_embedder()
    improved = []
    changed = 0
    answer_rows = 0
    for row in rows:
        proposed_hits = row.get("proposed_hits") or []
        proposed_decision = row.get("proposed_decision", "")
        proposed_answer = row.get("proposed_answer", "")
        proposed_citations = proposed_hits[:1] if proposed_hits and proposed_decision == "ANSWER" else []
        if proposed_decision == "ANSWER" and proposed_hits:
            answer_rows += 1
            synthesized = synthesize_extractive_answer(row.get("query", ""), proposed_hits[:1])
            if synthesized.get("status") == "ok" and synthesized.get("answer"):
                evidence_text = str(proposed_hits[0].get("text", ""))
                original_score = _score_answer(
                    row.get("query", ""),
                    proposed_answer,
                    proposed_citations,
                    evidence_text,
                    embedder,
                )
                synthesized_score = _score_answer(
                    row.get("query", ""),
                    str(synthesized["answer"]),
                    proposed_citations,
                    evidence_text,
                    embedder,
                )
                original_rank = (int(original_score["esa_pass"]), float(original_score["aqs"]))
                synthesized_rank = (int(synthesized_score["esa_pass"]), float(synthesized_score["aqs"]))
                if synthesized_rank > original_rank:
                    proposed_answer = str(synthesized["answer"])
                    changed += 1
        improved.append(
            {
                "query_id": row.get("query_id"),
                "query": row.get("query", ""),
                "history": row.get("history", []),
                "gold_doc_id": row.get("gold_doc_id"),
                "gold_chunk_id": row.get("gold_chunk_id"),
                "gold_domain": row.get("gold_domain"),
                "gold_triage": row.get("gold_triage"),
                "gold_answer": row.get("gold_answer", ""),
                "reference_answer": row.get("reference_answer", ""),
                "baseline_decision": row.get("baseline_0_pretrained_decision", "ANSWER"),
                "baseline_hits": row.get("baseline_0_pretrained_hits", []),
                "baseline_answer": row.get("baseline_0_pretrained_answer")
                or cited_answer(row.get("query", ""), row.get("baseline_0_pretrained_hits", [])),
                "proposed_decision": proposed_decision,
                "proposed_hits": proposed_hits,
                "proposed_citations": proposed_citations,
                "proposed_answer": proposed_answer,
            }
        )
    write_jsonl(OUTPUT, improved)
    SUMMARY.write_text(
        "\n".join(
            [
                "# Supported Synthesis Answer Improvement",
                "",
                f"Input: `{INPUT}`",
                f"Output: `{OUTPUT}`",
                "",
                "This pass does not change retrieved hits, citations, routing, triage decisions, tickets, or rejects.",
                "It only rewrites Proposed `ANSWER` text from the already selected top evidence passage using the extractive synthesizer.",
                "",
                f"Rows: `{len(rows)}`",
                f"Proposed ANSWER rows: `{answer_rows}`",
                f"ANSWER text rewritten: `{changed}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "answer_rows": answer_rows, "rewritten": changed, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
