from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluate_end_to_end import run_proposed
from src.presentation import format_tool_trace, presentation_result, reject_from_trace, ticket_from_trace
from src.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proposed_final.yaml")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    query = args.query or input("User query: ")
    result = presentation_result(run_proposed(query, config), query, config)
    print("User query:")
    print(query)
    print("\nDecision:")
    print(result.get("decision"))
    print()
    print(format_tool_trace(result.get("tool_trace", [])))
    print("\nFinal answer:")
    display_answer = result.get("display_answer") or result.get("answer", "")
    if result.get("decision") == "TICKET":
        display_answer = display_answer.split("Ticket ID:")[0].strip()
    print(display_answer)
    generator = result.get("generator") or {}
    if generator:
        model_name = str(generator.get("model_name") or "")
        label = "extractive fallback" if "extractive" in model_name or generator.get("fallback_reason") else "flan-t5-fixed"
        print("\nGenerator:")
        print(f"{label} ({model_name or 'not used'})")
    if result.get("decision") == "ANSWER":
        print("\nCitations:")
        citations = result.get("citations") or []
        if not citations:
            print("none")
        for citation in citations:
            print(f"- doc_id: {citation.get('doc_id')}")
            print(f"  chunk_id: {citation.get('chunk_id')}")
            print(f"  span: {citation.get('span_start')}-{citation.get('span_end')}")
    elif result.get("decision") == "TICKET":
        ticket = ticket_from_trace(result.get("tool_trace", []))
        if ticket:
            print("\nTicket:")
            print(f"- ticket_id: {ticket.get('ticket_id', 'not available')}")
            print(f"- category: {ticket.get('category', 'support')}")
            print(f"- severity: {ticket.get('severity', 'medium')}")
    elif result.get("decision") == "REJECT":
        reject = reject_from_trace(result.get("tool_trace", []))
        if reject:
            print("\nReject reason:")
            print(reject.get("reason", "out_of_domain"))
    if result.get("latency_ms") is not None:
        print("\nLatency:")
        print(f"{float(result['latency_ms']):.2f} ms")


if __name__ == "__main__":
    main()
