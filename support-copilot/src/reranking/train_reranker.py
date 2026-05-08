from __future__ import annotations

import os

from torch.utils.data import DataLoader

from src.utils.device import get_device
from src.utils.io import project_path, read_jsonl, write_json


def train_reranker(config: dict) -> dict:
    rows = read_jsonl(project_path("data", "processed", "reranker_train.jsonl"))
    if config.get("mode") in {"full_local", "full"} and rows:
        limit = config.get("max_reranker_train_pairs")
        if limit:
            rows = rows[: int(limit)]
        os.environ.setdefault("HF_HOME", str(project_path("data", "hf_cache", "hub")))
        from sentence_transformers import CrossEncoder, InputExample

        chunks = {row["chunk_id"]: row["text"] for row in read_jsonl(project_path("data", "processed", "kb_chunks.jsonl"))}
        examples = [
            InputExample(texts=[row["query"], chunks[row["chunk_id"]]], label=float(row["label"]))
            for row in rows
            if row.get("chunk_id") in chunks
        ]
        model_name = str(config.get("reranker_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        out_dir = project_path("outputs", "reranker", "cross_encoder")
        device = get_device(str(config.get("device", "auto")))
        model = CrossEncoder(model_name, num_labels=1, device=device)
        if examples:
            loader = DataLoader(examples, shuffle=True, batch_size=int(config.get("reranker_batch_size", 4)))
            model.fit(
                train_dataloader=loader,
                epochs=int(config.get("reranker_epochs", 1)),
                warmup_steps=0,
                output_path=str(out_dir),
                save_best_model=False,
                use_amp=bool(config.get("fp16", False) and device == "cuda"),
                show_progress_bar=False,
            )
            model.save(str(out_dir))
        else:
            model.save(str(out_dir))
        positives = sum(1 for r in rows if r.get("label") == 1)
        metrics = {
            "trained_pairs": len(examples),
            "positives": positives,
            "model": model_name,
            "backend": "cross-encoder",
            "checkpoint": str(out_dir.relative_to(project_path())),
            "device": str(model.model.device),
        }
        write_json(project_path("outputs", "reranker", "model.json"), metrics)
        return metrics
    positives = sum(1 for r in rows if r.get("label") == 1)
    metrics = {"trained_pairs": len(rows), "positives": positives, "model": "lexical-cross-encoder-lite"}
    write_json(project_path("outputs", "reranker", "model.json"), metrics)
    return metrics
