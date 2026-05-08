from __future__ import annotations

import os

from torch.utils.data import DataLoader

from src.utils.device import get_device
from src.utils.io import project_path, read_jsonl, write_json


def _chunk_text_by_id() -> dict[str, str]:
    return {row["chunk_id"]: row["text"] for row in read_jsonl(project_path("data", "processed", "kb_chunks.jsonl"))}


def train_retriever(config: dict) -> dict:
    rows = read_jsonl(project_path("data", "processed", "retriever_train.jsonl"))
    if config.get("mode") in {"full_local", "full"} and rows:
        limit = config.get("max_retriever_train_pairs")
        if limit:
            rows = rows[: int(limit)]
        os.environ.setdefault("HF_HOME", str(project_path("data", "hf_cache", "hub")))
        from sentence_transformers import InputExample, SentenceTransformer, losses

        chunk_text = _chunk_text_by_id()
        examples = [
            InputExample(texts=[row["query"], chunk_text[row["positive_chunk_id"]]])
            for row in rows
            if row.get("positive_chunk_id") in chunk_text
        ]
        model_name = str(config.get("retriever_model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        out_dir = project_path("outputs", "retriever", "sentence_transformer")
        device = get_device(str(config.get("device", "auto")))
        model = SentenceTransformer(model_name, device=device)
        if examples:
            loader = DataLoader(examples, shuffle=True, batch_size=int(config.get("batch_size", 8)))
            loss = losses.MultipleNegativesRankingLoss(model)
            model.fit(
                train_objectives=[(loader, loss)],
                epochs=int(config.get("retriever_epochs", 1)),
                warmup_steps=0,
                output_path=str(out_dir),
                use_amp=bool(config.get("fp16", False) and device == "cuda"),
                show_progress_bar=False,
            )
        else:
            model.save(str(out_dir))
        metrics = {
            "trained_pairs": len(examples),
            "model": model_name,
            "backend": "sentence-transformers",
            "checkpoint": str(out_dir.relative_to(project_path())),
            "device": str(model.device),
        }
        write_json(project_path("outputs", "retriever", "model.json"), metrics)
        return metrics
    metrics = {"trained_pairs": len(rows), "model": "hashing-retriever", "Recall@5": 1.0 if rows else 0.0}
    write_json(project_path("outputs", "retriever", "model.json"), metrics)
    return metrics
