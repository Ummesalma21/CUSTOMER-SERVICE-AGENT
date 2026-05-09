from __future__ import annotations

from functools import lru_cache

import re

from src.generation.answer_quality import clean_answer_text, is_fragment_answer
from src.generation.extractive_synthesizer import synthesize_extractive_answer
from src.retrieval.search_kb import tokenize


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
                fallback = synthesize_extractive_answer(query, evidence_passages)
                fallback["fallback_reason"] = "insufficient_evidence_token"
                return fallback
            if _invalid_generation(answer, query, evidence_passages):
                fallback = synthesize_extractive_answer(query, evidence_passages)
                fallback["fallback_reason"] = "invalid_generation"
                return fallback
            extractive = synthesize_extractive_answer(query, evidence_passages)
            if _should_prefer_extractive(query, answer, extractive, evidence_passages):
                extractive["fallback_reason"] = "extractive_more_evidence_supported"
                return extractive
            return {
                "status": "ok",
                "answer": answer,
                "used_evidence_ids": [str(p.get("chunk_id", "")) for p in evidence_passages[:2]],
                "model_name": candidate_model,
                "fallback_reason": None,
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


def _invalid_generation(answer: str, query: str, evidence_passages: list[dict]) -> bool:
    if not answer:
        return True
    if is_fragment_answer(answer):
        return True
    if not re.search(r"[.!?]\s*$", answer.strip()) and len(_tokens(answer)) > 40:
        return True
    if len(_tokens(answer)) > 70:
        return True
    if answer.strip().endswith("?"):
        return True
    if _has_repeated_sentence(answer):
        return True
    a = _tokens(answer)
    q = _tokens(query)
    if a and q and len(set(a) & set(q)) / max(1, len(set(a))) > 0.85:
        return True
    q_content = {tok for tok in q if len(tok) >= 4 and tok not in _QUERY_STOPWORDS}
    if len(q_content) >= 2 and len(q_content & set(a)) / max(1, len(q_content)) < 0.5:
        return True
    if re.search(r"\b[a-z]{10,}(and|with|or|for|to)[a-z]{4,}\b", answer.lower()):
        return True
    evidence_tokens = {tok for passage in evidence_passages for tok in _tokens(str(passage.get("text", "")))}
    content = {tok for tok in a if len(tok) >= 4}
    if content and evidence_tokens and len(content & evidence_tokens) / max(1, len(content)) < 0.25:
        return True
    return False


def _should_prefer_extractive(query: str, generated_answer: str, extractive: dict, evidence_passages: list[dict]) -> bool:
    if extractive.get("status") != "ok" or not extractive.get("answer"):
        return False
    generated = clean_answer_text(generated_answer)
    extracted = clean_answer_text(str(extractive.get("answer", "")))
    evidence = " ".join(str(p.get("text", "")) for p in evidence_passages[:2])
    generated_support = _support_score(query, generated, evidence)
    extractive_support = _support_score(query, extracted, evidence)
    if _invalid_generation(generated, query, evidence_passages):
        return True
    if extractive_support >= generated_support + 0.08:
        return True
    if generated_support < 0.42 and extractive_support >= 0.42:
        return True
    return False


def _support_score(query: str, answer: str, evidence: str) -> float:
    q_tokens = _content_tokens(query)
    a_tokens = _content_tokens(answer)
    e_tokens = _content_tokens(evidence)
    if not a_tokens or not e_tokens:
        return 0.0
    answer_evidence = len(a_tokens & e_tokens) / max(1, len(a_tokens))
    query_answer = len(q_tokens & a_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
    query_evidence = len(q_tokens & e_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
    return 0.55 * answer_evidence + 0.30 * query_answer + 0.15 * query_evidence


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower())


def _content_tokens(text: str) -> set[str]:
    return {tok for tok in tokenize(text) if len(tok) >= 3 and tok not in _QUERY_STOPWORDS}


_QUERY_STOPWORDS = {
    "about",
    "after",
    "before",
    "could",
    "does",
    "have",
    "help",
    "here",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def _has_repeated_sentence(answer: str) -> bool:
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", answer or "") if s.strip()]
    if len(sentences) != len(set(sentences)):
        return True
    words = _tokens(answer)
    if len(words) >= 18:
        chunks = [" ".join(words[i : i + 8]) for i in range(0, len(words) - 7)]
        if len(chunks) != len(set(chunks)):
            return True
    return False
