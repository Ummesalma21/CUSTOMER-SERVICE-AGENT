from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import project_path, read_json, write_json


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    baseline_src = read_json(project_path("outputs", "reports", "baseline_vs_proposed_metrics.json"), {})
    proposed_src = read_json(project_path("outputs", "reports", "domain_router_fallback_grid_fixed_metrics.json"), {})

    baseline = {
        "answer_only": (baseline_src.get("answer_only") or {}).get("baseline", {}),
        "mixed_workflow": (baseline_src.get("mixed_workflow") or {}).get("baseline", {}),
        "unsupported_answer_safety": (baseline_src.get("unsupported_answer_safety") or {}).get("baseline", {}),
    }
    proposed = {
        "selected_validation_setting": proposed_src.get("selected_validation_setting", {}),
        "test_metrics": proposed_src.get("test_metrics", {}),
    }

    out_dir = project_path("outputs", "final")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "baseline_final_metrics.json", baseline)
    write_json(out_dir / "proposed_domain_router_final_metrics.json", proposed)

    answer_base = baseline["answer_only"]
    answer_prop = (proposed["test_metrics"] or {}).get("answer_only", {})
    mixed_base = baseline["mixed_workflow"]
    mixed_prop = (proposed["test_metrics"] or {}).get("mixed_workflow", {})
    safe_base = baseline["unsupported_answer_safety"]
    safe_prop = (proposed["test_metrics"] or {}).get("unsupported_answer_safety", {})
    setting = proposed["selected_validation_setting"]

    baseline_esa = answer_base.get("ESA", 0.4760)
    baseline_aqs = answer_base.get("AQS", 0.6270)

    lines = [
        "# Final Comparison: Baseline vs Proposed Domain-Router",
        "",
        "## Selected Proposed Thresholds",
        "",
        f"- top_k_domains: `{setting.get('top_k_domains')}`",
        f"- min_domain_confidence: `{setting.get('min_domain_confidence')}`",
        f"- min_candidate_similarity: `{setting.get('min_candidate_similarity')}`",
        f"- min_domain_candidates: `{setting.get('min_domain_candidates')}`",
        f"- fallback_merge_mode: `{setting.get('fallback_merge_mode')}`",
        f"- rerank_after_merge: `{setting.get('rerank_after_merge')}`",
        "",
        "## Final Metrics",
        "",
        "| Metric | Baseline | Proposed (Domain-Router) |",
        "|---|---:|---:|",
        f"| Recall@5 | {_fmt(answer_base.get('Recall@5'))} | {_fmt(answer_prop.get('Recall@5'))} |",
        f"| EvidenceHit@5 | {_fmt(answer_base.get('EvidenceHit@5'))} | {_fmt(answer_prop.get('EvidenceHit@5'))} |",
        f"| ESA | {_fmt(baseline_esa)} | {_fmt(answer_prop.get('ESA'))} |",
        f"| AQS | {_fmt(baseline_aqs)} | {_fmt(answer_prop.get('AQS'))} |",
        f"| Macro-F1 | {_fmt(mixed_base.get('Macro-F1'))} | {_fmt(mixed_prop.get('Macro-F1'))} |",
        f"| UnsupportedAnswerRate | {_fmt(safe_base.get('UnsupportedAnswerRate'))} | {_fmt(safe_prop.get('UnsupportedAnswerRate'))} |",
        f"| OODAnswerRate | {_fmt(safe_base.get('OODAnswerRate'))} | {_fmt(safe_prop.get('OODAnswerRate'))} |",
        f"| TicketMissRate | {_fmt(safe_base.get('TicketMissRate'))} | {_fmt(safe_prop.get('TicketMissRate'))} |",
    ]
    (out_dir / "final_comparison_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"baseline_out": str(out_dir / "baseline_final_metrics.json"), "proposed_out": str(out_dir / "proposed_domain_router_final_metrics.json"), "table_out": str(out_dir / "final_comparison_table.md")}, indent=2))


if __name__ == "__main__":
    main()
