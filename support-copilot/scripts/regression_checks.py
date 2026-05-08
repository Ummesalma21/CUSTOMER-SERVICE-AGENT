from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_end_to_end import run_proposed
from src.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    benefits = run_proposed("Can I renew my benefits online?", config)
    if benefits.get("decision") != "ANSWER" or not benefits.get("citations"):
        raise SystemExit(f"benefits regression failed: expected ANSWER with citation, got {benefits}")
    evidence = " ".join(
        " ".join(str(c.get(k, "")) for k in ("doc_id", "title", "text")).lower()
        for c in benefits.get("citations", []) + benefits.get("hits", [])[:1]
    )
    if not ("benefit" in evidence and ("renew" in evidence or "renewal" in evidence)):
        raise SystemExit(f"benefits regression failed: citation was not benefits/renewal evidence: {benefits}")

    ipl = run_proposed("Who won the IPL yesterday?", config)
    trace_names = [t.get("name") or t.get("tool") for t in ipl.get("tool_trace", [])]
    if ipl.get("decision") != "REJECT" or "RejectQuery" not in trace_names:
        raise SystemExit(f"reject regression failed: expected RejectQuery, got {ipl}")

    print({"benefits_decision": benefits["decision"], "benefits_citation": benefits["citations"][:1], "ipl_trace": trace_names})


if __name__ == "__main__":
    main()
