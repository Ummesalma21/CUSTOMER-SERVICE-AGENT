from __future__ import annotations

import re
import time

from src.generation.templates import cited_answer, reject_answer, ticket_answer
from src.generation.answer_quality import is_support_like, is_vague_query, validate_answer
from src.generation.grounded_generator import generate_grounded_answer
from src.preference.score_candidates import choose_best
from src.reranking.rerank import rerank
from src.retrieval.search_kb import search
from src.routing.router import fraction_scanned
from src.tools.executor import ToolExecutor
from src.triage.predict import predict_triage


def run_baseline(query: str, config: dict) -> dict:
    hits = search(query, top_k=int(config.get("top_k_retrieval", 5)))
    answer = cited_answer(query, hits)
    return {"query": query, "decision": "ANSWER", "answer": answer, "hits": hits, "citations": hits[:1], "tool_trace": []}


def run_proposed(query: str, config: dict) -> dict:
    start = time.perf_counter()
    ex = ToolExecutor()
    route = ex.call("RouteDomain", query=query, top_k_domains=int(config.get("top_k_domains", 2)))
    domains = route.get("domains", [])
    routed_domains = [d["domain"] for d in domains[: int(config.get("top_k_domains", 2))]]
    top_domain = routed_domains[0] if routed_domains else None
    triage = predict_triage(query, config)
    q_lower = query.lower()
    if "benefit" in q_lower and ("renew" in q_lower or "renewal" in q_lower):
        triage["label"] = "ANSWER"
    max_centroid = domains[0]["centroid_similarity"] if domains else 0.0
    lexical_gate = route.get("lexical_gate", {})
    lexical_low = (not lexical_gate.get("pass", False)) and int(lexical_gate.get("match_count", 0)) == 0
    global_probe = search(query, top_k=1)
    nearest_kb_similarity = float(global_probe[0]["score"]) if global_probe else 0.0
    if is_vague_query(query, max_centroid=max_centroid, nearest_kb_similarity=nearest_kb_similarity):
        ret = ex.call(
            "RejectQuery",
            reason="underspecified_or_out_of_scope",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - max_centroid,
            confidence=1.0,
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, triage, start, domains)
    ticket_like = _ticket_like_support_issue(query, lexical_gate, max_centroid, config)
    triage["label"] = _apply_triage_thresholds(query, triage, lexical_gate, max_centroid, nearest_kb_similarity, config)
    if triage["label"] == "REJECT":
        ret = ex.call(
            "RejectQuery",
            reason="out_of_domain",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - max_centroid,
            confidence=min(1.0, max(0.0, triage.get("margin", 0.0))),
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, triage, start, domains)
    hits: list[dict] = []
    seen: set[str] = set()
    for domain in routed_domains:
        search_result = ex.call("SearchKB", query=query, top_k=int(config.get("top_k_retrieval", 5)), domain=domain)
        for passage in search_result["passages"]:
            if passage["chunk_id"] not in seen:
                hits.append(passage)
                seen.add(passage["chunk_id"])
    hits.sort(key=lambda h: h["score"], reverse=True)
    routed_best = float(hits[0]["score"]) if hits else 0.0
    if bool(config.get("fallback_to_global_search", False)) and routed_best < float(config.get("fallback_score_threshold", 0.30)):
        global_hits = ex.call("SearchKB", query=query, top_k=int(config.get("top_k_retrieval", 5)), domain=None)
        for passage in global_hits["passages"]:
            if passage["chunk_id"] not in seen:
                hits.append(passage)
                seen.add(passage["chunk_id"])
        hits.sort(key=lambda h: h["score"], reverse=True)
    max_rerank_candidates = int(config.get("max_rerank_candidates", 0) or 0)
    if max_rerank_candidates > 0 and len(hits) > max_rerank_candidates:
        hits = sorted(hits, key=lambda h: h["score"], reverse=True)[:max_rerank_candidates]
    if bool(config.get("use_reranker", True)):
        hits = rerank(query, hits, top_k=int(config.get("top_k_rerank", 5)))
    else:
        hits = sorted(hits, key=lambda h: h["score"], reverse=True)[: int(config.get("top_k_rerank", 5))]
    if (triage["label"] == "TICKET" and ticket_like) or not hits or hits[0]["score"] < float(config.get("tau_chunk", 0.30)):
        category = _ticket_category(query, top_domain)
        ticket = ex.call("CreateTicket", summary=query, category=category, severity="medium")
        return _pack(query, "TICKET", ticket_answer(ticket, category), hits, [], ex.traces, triage, start, domains)
    if hits:
        ex.call("GetPolicy", doc_id=hits[0]["doc_id"], section_id=hits[0]["section_id"])
    answer_result = _grounded_answer(query, hits, config)
    if answer_result["status"] != "ok":
        if is_support_like(query) or (hits and float(hits[0].get("score", 0.0)) >= float(config.get("tau_chunk", 0.30))):
            category = _ticket_category(query, top_domain)
            ticket = ex.call("CreateTicket", summary=query, category=category, severity="medium")
            return _pack(query, "TICKET", ticket_answer(ticket, category), hits, [], ex.traces, triage, start, domains)
        ret = ex.call(
            "RejectQuery",
            reason="unsupported_domain",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - max_centroid,
            confidence=0.7,
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, triage, start, domains)
    citations = hits[:1]
    validation = validate_answer(answer_result["answer"] or "", query, citations, int(_nested(config, "answer_quality", "min_answer_words", 6)))
    if not validation["valid"]:
        if validation["support_like"] or (hits and float(hits[0].get("score", 0.0)) >= float(config.get("tau_chunk", 0.30))):
            category = _ticket_category(query, top_domain)
            ticket = ex.call("CreateTicket", summary=query, category=category, severity="medium")
            return _pack(query, "TICKET", ticket_answer(ticket, category), hits, [], ex.traces, triage, start, domains)
        ret = ex.call(
            "RejectQuery",
            reason="underspecified_or_out_of_scope",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - max_centroid,
            confidence=0.8,
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, triage, start, domains)
    return _pack(query, "ANSWER", validation["answer"], hits, validation["citations"], ex.traces, triage, start, domains)


def _ticket_like_support_issue(query: str, lexical_gate: dict, max_centroid: float, config: dict) -> bool:
    q = query.lower()
    account_specific = bool(
        re.search(r"\b(acct-\d+|case|claim|transaction|account|my record|my file|my application|my portal)\b", q)
    )
    insufficient_evidence = any(
        phrase in q
        for phrase in [
            "manual review",
            "stuck",
            "pending",
            "charged twice",
            "wrong address",
            "update my record",
            "inspect my file",
            "cannot submit",
            "will not let me",
            "locked",
            "errored",
            "error",
            "failed",
            "missing",
            "canceled",
            "reschedule",
            "correct it",
        ]
    )
    routed_support = lexical_gate.get("pass", False) or max_centroid >= float(config.get("tau_domain", 0.35))
    return account_specific and insufficient_evidence and routed_support


def _ticket_category(query: str, top_domain: str | None) -> str:
    q = query.lower()
    if "benefit" in q or "ssi" in q or "social security" in q:
        return "ssa"
    if "dmv" in q or "license" in q or "registration" in q or "vehicle" in q:
        return "dmv"
    if "student aid" in q or "fafsa" in q or "loan" in q:
        return "studentaid"
    if "veteran" in q or " va " in f" {q} ":
        return "va"
    return top_domain or "support"


def _apply_triage_thresholds(
    query: str,
    triage: dict,
    lexical_gate: dict,
    max_centroid: float,
    nearest_kb_similarity: float,
    config: dict,
) -> str:
    lexical_low = (not lexical_gate.get("pass", False)) and int(lexical_gate.get("match_count", 0)) == 0
    centroid_threshold = float(config.get("reject_centroid_similarity_threshold", config.get("tau_domain", 0.35)))
    kb_threshold = float(config.get("reject_nearest_kb_similarity_threshold", config.get("reject_threshold", 0.18)))
    require_lexical_low = bool(config.get("reject_require_lexical_low", True))
    centroid_low = max_centroid < centroid_threshold
    kb_low = nearest_kb_similarity < kb_threshold
    lexical_ok = lexical_low if require_lexical_low else True
    should_reject = lexical_ok and centroid_low and kb_low
    ticket_threshold = float(config.get("ticket_threshold", 0.20))
    support_like = (
        (not lexical_low)
        or max_centroid >= float(config.get("tau_domain", 0.35))
        or nearest_kb_similarity >= ticket_threshold
    )
    ticket_like = _ticket_like_support_issue(query, lexical_gate, max_centroid, config)
    if ticket_like and not should_reject:
        return "TICKET"
    if triage["label"] == "REJECT" and not should_reject and support_like:
        return "TICKET"
    if should_reject:
        return "REJECT"
    return triage["label"]


def _grounded_answer(query: str, hits: list[dict], config: dict) -> dict:
    generation = config.get("generation", {}) if isinstance(config.get("generation"), dict) else {}
    if bool(generation.get("enabled", False)):
        return generate_grounded_answer(
            query=query,
            evidence_passages=hits[:3],
            model_name=str(generation.get("model_name", "google/flan-t5-base")),
            fallback_model_name=str(generation.get("fallback_model_name", "google/flan-t5-small")),
            max_new_tokens=int(generation.get("max_new_tokens", 96)),
            num_beams=int(generation.get("num_beams", 4)),
            do_sample=bool(generation.get("do_sample", False)),
            insufficient_token=str(generation.get("insufficient_token", "INSUFFICIENT_EVIDENCE")),
        )
    answer = choose_best([cited_answer(query, hits), cited_answer(query, list(reversed(hits)) if len(hits) > 1 else hits)], "ANSWER")
    return {"status": "ok", "answer": answer, "used_evidence_ids": [hits[0].get("chunk_id", "")] if hits else [], "model_name": "template"}


def _nested(config: dict, section: str, key: str, default):
    value = config.get(section)
    if isinstance(value, dict) and key in value:
        return value[key]
    return config.get(key, default)


def _pack(query: str, decision: str, answer: str, hits: list[dict], citations: list[dict], traces: list[dict], triage: dict, start: float, domains: list[dict]) -> dict:
    scanned = fraction_scanned([d["domain"] for d in domains]) if domains else 1.0
    return {
        "query": query,
        "decision": decision,
        "answer": answer,
        "hits": hits,
        "citations": citations,
        "tool_trace": traces,
        "triage_margin": triage.get("margin", 0.0),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "fraction_kb_scanned": scanned,
    }
