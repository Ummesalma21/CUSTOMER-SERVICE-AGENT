from __future__ import annotations

import re
import time

from src.generation.templates import cited_answer, reject_answer, ticket_answer
from src.generation.answer_quality import is_support_like, is_vague_query, validate_answer
from src.generation.grounded_generator import generate_grounded_answer
from src.policy.answerability import should_reject as policy_should_reject
from src.policy.answerability import should_ticket as policy_should_ticket
from src.preference.score_candidates import choose_best
from src.reranking.rerank import rerank
from src.retrieval.search_kb import search
from src.routing.domain_router import retrieve_primary_domains, route_query
from src.routing.router import fraction_scanned
from src.tools.executor import ToolExecutor


def run_baseline(query: str, config: dict) -> dict:
    hits = search(query, top_k=int(config.get("top_k_retrieval", 5)))
    answer = cited_answer(query, hits)
    return {"query": query, "decision": "ANSWER", "answer": answer, "hits": hits, "citations": hits[:1], "tool_trace": []}


def run_proposed(query: str, config: dict) -> dict:
    start = time.perf_counter()
    ex = ToolExecutor()
    routed = route_query(query, None, config)
    routed_domains = routed.get("kb_domains", [])
    top_domain = routed_domains[0] if routed_domains else None
    action = (routed.get("action") or {}).get("label", "ANSWER")
    action_confidence = float((routed.get("action") or {}).get("confidence", 0.0))

    primary_hits, retrieval_info = retrieve_primary_domains(query, routed, config)
    hits: list[dict] = list(primary_hits)
    if retrieval_info.get("fallback_triggered", False):
        global_hits = ex.call("SearchKB", query=query, top_k=int(config.get("top_k_retrieval", 20)), domain=None)
        hits = _merge_hits(hits + list(global_hits.get("passages", [])))
    else:
        for domain in routed_domains:
            domain_hits = ex.call("SearchKB", query=query, top_k=int(config.get("top_k_retrieval", 20)), domain=domain)
            hits = _merge_hits(hits + list(domain_hits.get("passages", [])))

    nearest_kb_similarity = float(hits[0]["score"]) if hits else 0.0
    if is_vague_query(query, max_centroid=action_confidence, nearest_kb_similarity=nearest_kb_similarity):
        ret = ex.call(
            "RejectQuery",
            reason="underspecified_or_out_of_scope",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - action_confidence,
            confidence=1.0,
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, {"label": "REJECT", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)

    if action == "REJECT":
        ret = ex.call(
            "RejectQuery",
            reason="out_of_domain",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - action_confidence,
            confidence=action_confidence,
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, {"label": "REJECT", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)

    if action == "TICKET":
        category = _ticket_category(query, top_domain)
        ticket = ex.call("CreateTicket", summary=query, category=category, severity="medium")
        return _pack(query, "TICKET", ticket_answer(ticket, category), hits, [], ex.traces, {"label": "TICKET", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)

    max_rerank_candidates = int(config.get("max_rerank_candidates", 0) or 0)
    if max_rerank_candidates > 0 and len(hits) > max_rerank_candidates:
        hits = sorted(hits, key=lambda h: h["score"], reverse=True)[:max_rerank_candidates]
    rerank_after_merge = bool(_nested(config, "domain_router", "rerank_after_merge", True))
    if rerank_after_merge and bool(config.get("use_reranker", False)):
        hits = rerank(query, hits, top_k=int(config.get("top_k_rerank", 5)))
    else:
        hits = sorted(hits, key=lambda h: h["score"], reverse=True)[:int(config.get("top_k_rerank", 5))]

    if not hits or hits[0]["score"] < float(config.get("tau_chunk", 0.30)):
        category = _ticket_category(query, top_domain)
        ticket = ex.call("CreateTicket", summary=query, category=category, severity="medium")
        return _pack(query, "TICKET", ticket_answer(ticket, category), hits, [], ex.traces, {"label": "TICKET", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)

    if hits:
        ex.call("GetPolicy", doc_id=hits[0]["doc_id"], section_id=hits[0]["section_id"])
    answer_result = _grounded_answer(query, hits, config)
    if answer_result["status"] != "ok":
        if is_support_like(query) or (hits and float(hits[0].get("score", 0.0)) >= float(config.get("tau_chunk", 0.30))):
            category = _ticket_category(query, top_domain)
            ticket = ex.call("CreateTicket", summary=query, category=category, severity="medium")
            return _pack(query, "TICKET", ticket_answer(ticket, category), hits, [], ex.traces, {"label": "TICKET", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)
        ret = ex.call(
            "RejectQuery",
            reason="unsupported_domain",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - action_confidence,
            confidence=0.7,
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, {"label": "REJECT", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)
    citations = hits[:1]
    validation = validate_answer(answer_result["answer"] or "", query, citations, int(_nested(config, "answer_quality", "min_answer_words", 6)))
    if not validation["valid"]:
        if validation["support_like"] or (hits and float(hits[0].get("score", 0.0)) >= float(config.get("tau_chunk", 0.30))):
            category = _ticket_category(query, top_domain)
            ticket = ex.call("CreateTicket", summary=query, category=category, severity="medium")
            return _pack(query, "TICKET", ticket_answer(ticket, category), hits, [], ex.traces, {"label": "TICKET", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)
        ret = ex.call(
            "RejectQuery",
            reason="underspecified_or_out_of_scope",
            nearest_kb_distance=1.0 - nearest_kb_similarity,
            nearest_centroid_distance=1.0 - action_confidence,
            confidence=0.8,
        )
        return _pack(query, "REJECT", ret["message"], [], [], ex.traces, {"label": "REJECT", "margin": action_confidence}, start, [], router_info=routed, retrieval_info=retrieval_info)
    return _pack(
        query,
        "ANSWER",
        validation["answer"],
        hits,
        validation["citations"],
        ex.traces,
        {"label": "ANSWER", "margin": action_confidence},
        start,
        [],
        generator_info=answer_result,
        router_info=routed,
        retrieval_info=retrieval_info,
    )


def _merge_hits(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for hit in sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True):
        key = hit.get("chunk_id")
        if key in seen:
            continue
        out.append(hit)
        seen.add(key)
    return out


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
    should_reject = policy_should_reject(
        query,
        lexical_low=lexical_low,
        centroid_similarity=max_centroid,
        nearest_kb_similarity=nearest_kb_similarity,
        centroid_threshold=centroid_threshold,
        kb_threshold=kb_threshold,
        require_lexical_low=require_lexical_low,
    )
    ticket_threshold = float(config.get("ticket_threshold", 0.20))
    support_like = (
        (not lexical_low)
        or max_centroid >= float(config.get("tau_domain", 0.35))
        or nearest_kb_similarity >= ticket_threshold
    )
    ticket_like = _ticket_like_support_issue(query, lexical_gate, max_centroid, config)
    if (ticket_like or policy_should_ticket(query, nearest_kb_similarity, ticket_threshold)) and not should_reject:
        return "TICKET"
    if triage["label"] == "REJECT" and not should_reject and support_like:
        return "TICKET"
    if should_reject:
        return "REJECT"
    return triage["label"]


def _grounded_answer(query: str, hits: list[dict], config: dict) -> dict:
    generation = config.get("generation", {}) if isinstance(config.get("generation"), dict) else {}
    if bool(generation.get("enabled", False)):
        top_evidence = max(1, min(3, int(generation.get("top_evidence_count", 1))))
        return generate_grounded_answer(
            query=query,
            evidence_passages=hits[:top_evidence],
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


def _pack(
    query: str,
    decision: str,
    answer: str,
    hits: list[dict],
    citations: list[dict],
    traces: list[dict],
    triage: dict,
    start: float,
    domains: list[dict],
    generator_info: dict | None = None,
    router_info: dict | None = None,
    retrieval_info: dict | None = None,
) -> dict:
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
        "generator": generator_info or {},
        "domain_router": router_info or {},
        "retrieval_info": retrieval_info or {},
    }
