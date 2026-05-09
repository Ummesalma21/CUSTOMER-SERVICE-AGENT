from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import project_path, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/generator_train.jsonl")
    parser.add_argument("--val", default="data/processed/generator_val.jsonl")
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=96)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = read_jsonl(project_path(*Path(args.train).parts))
    val_rows = read_jsonl(project_path(*Path(args.val).parts))
    diagnostics = _diagnose(rows, tokenizer, args)
    val_diagnostics = _diagnose(val_rows, tokenizer, args)
    report = _markdown(args, diagnostics, val_diagnostics)
    out = project_path("outputs", "reports", "generator_batch_debug.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    write_jsonl(project_path("outputs", "reports", "generator_batch_debug_examples.jsonl"), diagnostics["examples"])
    print(report)


def _diagnose(rows: list[dict], tokenizer, args) -> dict:
    examples = []
    all_ignored = 0
    total = 0
    label_tokens = []
    label_minus100 = []
    invalid_ids = 0
    for batch_idx in range(args.num_batches):
        batch_rows = rows[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        if not batch_rows:
            break
        batch = [_encode(row, tokenizer, args) for row in batch_rows]
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        for i, item in enumerate(batch):
            non_ignored = int((labels[i] != -100).sum().item())
            ignored = int((labels[i] == -100).sum().item())
            total += 1
            label_tokens.append(non_ignored)
            label_minus100.append(ignored / max(1, labels.shape[1]))
            all_ignored += int(non_ignored == 0)
            invalid_ids += int(bool((input_ids[i] < 0).any() or (input_ids[i] >= tokenizer.vocab_size).any()))
            decoded_label_ids = [tok for tok in labels[i].tolist() if tok != -100]
            examples.append(
                {
                    "batch": batch_idx,
                    "row_index": batch_idx * args.batch_size + i,
                    "input_shape": list(input_ids.shape),
                    "attention_mask_shape": list(attention_mask.shape),
                    "labels_shape": list(labels.shape),
                    "label_minus100_rate": ignored / max(1, labels.shape[1]),
                    "non_ignored_label_tokens": non_ignored,
                    "all_labels_ignored": non_ignored == 0,
                    "decoded_input": tokenizer.decode(input_ids[i], skip_special_tokens=True)[:1500],
                    "decoded_label": tokenizer.decode(decoded_label_ids, skip_special_tokens=True),
                    "target": item["target"],
                    "target_words": len(str(item["target"]).split()),
                    "target_tokens_before_mask": item["target_tokens_before_mask"],
                    "has_nan_or_invalid_ids": False,
                }
            )
    return {
        "rows_checked": total,
        "all_ignored_labels": all_ignored,
        "all_ignored_label_rate": all_ignored / max(1, total),
        "avg_non_ignored_label_tokens": sum(label_tokens) / max(1, len(label_tokens)),
        "avg_label_minus100_rate": sum(label_minus100) / max(1, len(label_minus100)),
        "invalid_input_id_examples": invalid_ids,
        "examples": examples,
    }


def _encode(row: dict, tokenizer, args) -> dict:
    source = row.get("prompt") or row.get("input") or ""
    target = row.get("target") or row.get("target_answer") or row.get("reference_answer") or ""
    source_enc = tokenizer(
        source,
        max_length=args.max_input_length,
        truncation=True,
        padding="max_length",
    )
    target_enc = tokenizer(
        text_target=target,
        max_length=args.max_target_length,
        truncation=True,
        padding="max_length",
    )
    labels = [tok if tok != tokenizer.pad_token_id else -100 for tok in target_enc["input_ids"]]
    return {
        "input_ids": source_enc["input_ids"],
        "attention_mask": source_enc["attention_mask"],
        "labels": labels,
        "target": target,
        "target_tokens_before_mask": sum(1 for tok in target_enc["input_ids"] if tok != tokenizer.pad_token_id),
    }


def _markdown(args, train_diag: dict, val_diag: dict) -> str:
    lines = [
        "# Generator Batch Debug",
        "",
        f"Model/tokenizer: `{args.model}`",
        f"Batch size checked: `{args.batch_size}`",
        f"Max input length: `{args.max_input_length}`",
        f"Max target length: `{args.max_target_length}`",
        "",
        "## Train",
        _diag_block(train_diag),
        "",
        "## Validation",
        _diag_block(val_diag),
        "",
        "Detailed decoded examples are saved to `outputs/reports/generator_batch_debug_examples.jsonl`.",
    ]
    return "\n".join(lines)


def _diag_block(diag: dict) -> str:
    return "\n".join(
        [
            f"- Rows checked: `{diag['rows_checked']}`",
            f"- All ignored labels: `{diag['all_ignored_labels']}`",
            f"- All ignored label rate: `{diag['all_ignored_label_rate']}`",
            f"- Average non-ignored label tokens: `{diag['avg_non_ignored_label_tokens']}`",
            f"- Average label -100 rate: `{diag['avg_label_minus100_rate']}`",
            f"- Invalid input ID examples: `{diag['invalid_input_id_examples']}`",
        ]
    )


if __name__ == "__main__":
    main()
