from __future__ import annotations

from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Streamlit is not installed. Run: .\\.venv\\Scripts\\python.exe -m pip install streamlit") from exc

from src.evaluation.evaluate_end_to_end import run_proposed
from src.presentation import format_tool_trace, presentation_result, reject_from_trace, ticket_from_trace
from src.utils.io import load_config


DEFAULT_CONFIG = "configs/final_eval_generator.yaml" if Path("configs/final_eval_generator.yaml").exists() else "configs/final_eval_balanced_triage_best.yaml"
EXAMPLES = [
    "Can I renew my benefits online?",
    "Who won the IPL yesterday?",
    "My benefits renewal is stuck as pending for case ACCT-555123; can someone check my account?",
    "Why am I here?",
]


def main() -> None:
    st.set_page_config(page_title="Customer Support Copilot", page_icon="?", layout="centered")
    st.title("Customer Support Copilot")
    st.caption(
        "Answers supported KB questions with citations, creates tickets for account-specific issues, "
        "and rejects unsupported out-of-domain questions."
    )
    config_path = st.sidebar.text_input("Config", DEFAULT_CONFIG)
    if not Path(config_path).exists():
        st.error(f"Config not found: {config_path}")
        return
    try:
        config = load_config(config_path)
    except Exception as exc:
        st.error(f"Could not load config: {exc}")
        return
    with st.expander("Example queries"):
        for example in EXAMPLES:
            st.write(f"- {example}")
    query = st.text_input("User query", value=st.session_state.get("query", ""))
    if not st.button("Run", type="primary"):
        return
    try:
        result = presentation_result(run_proposed(query, config), query, config)
    except FileNotFoundError as exc:
        st.error(f"Missing checkpoint or data file: {exc}")
        return
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return
    decision = result.get("decision", "UNKNOWN")
    st.subheader(f"Decision: {decision}")
    if result.get("latency_ms") is not None:
        st.caption(f"Latency: {float(result['latency_ms']):.2f} ms")
    st.markdown("### Final answer")
    display_answer = result.get("display_answer") or result.get("answer", "")
    if decision == "TICKET":
        display_answer = display_answer.split("Ticket ID:")[0].strip()
    st.info(display_answer)
    if decision == "ANSWER":
        st.markdown("### Citations")
        citations = result.get("citations") or []
        if not citations:
            st.write("none")
        for citation in citations:
            st.write(
                {
                    "doc_id": citation.get("doc_id"),
                    "chunk_id": citation.get("chunk_id"),
                    "span": f"{citation.get('span_start')}-{citation.get('span_end')}",
                }
            )
    elif decision == "TICKET":
        ticket = ticket_from_trace(result.get("tool_trace", []))
        st.markdown("### Ticket")
        st.write(ticket or {"ticket_id": "not available"})
    elif decision == "REJECT":
        reject = reject_from_trace(result.get("tool_trace", []))
        st.markdown("### Reject reason")
        st.write((reject or {}).get("reason", "out_of_domain"))
    with st.expander("Tool trace"):
        st.code(format_tool_trace(result.get("tool_trace", [])))


if __name__ == "__main__":
    main()
