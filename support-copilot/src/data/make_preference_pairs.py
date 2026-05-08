from __future__ import annotations


def make_preference_pairs(eval_rows: list[dict], chunks_by_id: dict[str, dict]) -> list[dict]:
    pairs: list[dict] = []
    for row in eval_rows:
        qid = row["query_id"]
        triage = row.get("gold_triage", "ANSWER")
        if triage == "ANSWER" and row.get("gold_chunk_id") in chunks_by_id:
            chunk = chunks_by_id[row["gold_chunk_id"]]
            cite = f"[doc_id={chunk['doc_id']}, chunk_id={chunk['chunk_id']}, span={chunk['span_start']}-{chunk['span_end']}]"
            preferred = f"{chunk['text'].split('.')[0].strip()}. {cite}"
            rejected = f"Sure, the answer is probably yes, but check the website for details."
        elif triage == "TICKET":
            preferred = "I could not find enough KB evidence to answer confidently, so I created a support ticket for human review."
            rejected = "This is definitely fixed already. No support ticket is needed."
        else:
            preferred = "I can only help with questions covered by this support knowledge base, so I cannot answer that here."
            rejected = "Here is an answer to your unrelated question without using the support KB."
        pairs.append({"query_id": qid, "query": row["query"], "preferred": preferred, "rejected": rejected})
    return pairs

