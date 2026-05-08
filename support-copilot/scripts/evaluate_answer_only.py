from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_end_to_end import run_baseline, run_proposed
from src.evaluation.metrics import grounding_metrics, retrieval_metrics
from src.utils.io import load_config, project_path, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_eval_balanced_triage.yaml")
    parser.add_argument("--eval-path", default="data/processed/eval_set.jsonl")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    start = time.perf_counter()
    config = load_config(args.config)
    rows = [r for r in read_jsonl(project_path(*Path(args.eval_path).parts)) if r.get("gold_chunk_id")]
    if args.max_rows:
        rows = rows[: args.max_rows]
    baseline = [run_baseline(r["query"], config) for r in rows]
    proposed = [run_proposed(r["query"], config) for r in rows]
    metrics = {
        "config": args.config,
        "eval_path": args.eval_path,
        "count": len(rows),
        "baseline": _answer_metrics(rows, baseline),
        "proposed": _answer_metrics(rows, proposed),
    }
    predictions = []
    for row, base, prop in zip(rows, baseline, proposed):
        predictions.append(
            {
                **row,
                "baseline_decision": base["decision"],
                "baseline_hits": base.get("hits", [])[:5],
                "proposed_decision": prop["decision"],
                "proposed_hits": prop.get("hits", [])[:5],
                "proposed_citations": prop.get("citations", []),
                "proposed_answer": prop.get("answer", ""),
            }
        )
    write_json(project_path("outputs", "reports", "final_answer_only_metrics.json"), metrics)
    write_jsonl(project_path("outputs", "reports", "final_answer_only_predictions.jsonl"), predictions)
    _write_summary(metrics)
    elapsed = time.perf_counter() - start
    project_path("outputs", "logs", "final_answer_only_eval.log").parent.mkdir(parents=True, exist_ok=True)
    project_path("outputs", "logs", "final_answer_only_eval.log").write_text(
        f"config={args.config}\neval_path={args.eval_path}\nrows={len(rows)}\nseconds={elapsed:.3f}\nmetrics={metrics}\n",
        encoding="utf-8",
    )
    print(metrics)


def _answer_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    out = retrieval_metrics(rows, predictions, k=5)
    out.update(grounding_metrics(predictions))
    return out


def _write_summary(metrics: dict) -> None:
    lines = [
        "# Final Answer-Only Evaluation",
        "",
        f"Config: `{metrics['config']}`",
        f"Eval file: `{metrics['eval_path']}`",
        f"Answerable rows: `{metrics['count']}`",
        "",
        "## Baseline RAG",
        f"`{metrics['baseline']}`",
        "",
        "## Proposed Balanced Triage",
        f"`{metrics['proposed']}`",
    ]
    project_path("outputs", "reports", "final_answer_only_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
