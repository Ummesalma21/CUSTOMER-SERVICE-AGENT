from __future__ import annotations

from functools import lru_cache

from src.generation.answer_quality import clean_answer_text, is_fragment_answer
from src.generation.extractive_synthesizer import synthesize_extractive_answer


def generate_grounded_answer(
    query: str,
    evidence_passages: list[dict],
    chat_history: list[dict] | None = None,
    model_name: str = "google/flan-t5-base",
    max_new_tokens: int = 96,
    num_beams: int = 4,
    do_sample: bool = False,
    fallback_model_name: str = "google/flan-t5-small",
    insufficient_token: str = "INSUFFICIENT_EVIDENCE",
) -> dict:
    if not evidence_passages:
        return {"status": "insufficient_evidence", "answer": None, "used_evidence_ids": [], "model_name": model_name}
    prompt = _prompt(query, evidence_passages, chat_history, insufficient_token)
    for candidate_model in [model_name, fallback_model_name]:
        try:
            generator = _load_generator(candidate_model)
            if generator is None:
                continue
            tokenizer, model = generator
            import torch

            enc = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
            answer = clean_answer_text(tokenizer.decode(out[0], skip_special_tokens=True))
            if answer.strip() == insufficient_token:
                return synthesize_extractive_answer(query, evidence_passages)
            if not answer or is_fragment_answer(answer):
                return synthesize_extractive_answer(query, evidence_passages)
            return {
                "status": "ok",
                "answer": answer,
                "used_evidence_ids": [str(p.get("chunk_id", "")) for p in evidence_passages[:2]],
                "model_name": candidate_model,
            }
        except Exception:
            continue
    fallback = synthesize_extractive_answer(query, evidence_passages)
    if fallback["status"] == "ok":
        fallback["model_name"] = "extractive_fallback_no_local_generator"
    return fallback


@lru_cache(maxsize=2)
def _load_generator(model_name: str):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return tokenizer, model
    except Exception:
        return None


def _prompt(query: str, evidence_passages: list[dict], chat_history: list[dict] | None, insufficient_token: str) -> str:
    evidence_lines = []
    for idx, passage in enumerate(evidence_passages[:3], start=1):
        evidence_lines.append(f"[{idx}] {passage.get('text', '')}")
    history = ""
    if chat_history:
        history = "\nChat history:\n" + "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history[-4:])
    return (
        f"Question:\n{query}\n"
        f"{history}\n\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
        + "\n\nInstruction:\n"
        "Answer the question using only the evidence above.\n"
        "Write one concise customer-support answer in complete sentences.\n"
        "Do not add facts not present in the evidence.\n"
        "Do not include citations or document IDs in the answer text.\n"
        f"If the evidence is insufficient, output exactly: {insufficient_token}"
    )
