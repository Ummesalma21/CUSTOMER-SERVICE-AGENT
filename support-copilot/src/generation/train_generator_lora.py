from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.io import ensure_dir, project_path, read_jsonl, write_json


def train_generator_lora(config: dict) -> dict:
    """Stable full-data seq2seq fine-tuning for the grounded generator.

    The function name remains for compatibility with the original scaffold. The
    implementation is a direct PyTorch loop so label masking, gradient clipping,
    and NaN/Inf checks are explicit and auditable.
    """
    gen_cfg = config.get("generator_training") if isinstance(config.get("generator_training"), dict) else {}
    seed = int(config.get("seed", gen_cfg.get("seed", 42)))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = str(gen_cfg.get("model_name", config.get("generator_model_name", "google/flan-t5-small")))
    train_file = project_path(str(gen_cfg.get("train_file", "data/processed/generator_train.jsonl")))
    val_file = project_path(str(gen_cfg.get("val_file", "data/processed/generator_val.jsonl")))
    output_dir = project_path(str(gen_cfg.get("output_dir", "outputs/generator/flan_t5_fixed")))
    metrics_file = project_path(str(gen_cfg.get("metrics_file", "outputs/reports/generator_fixed_train_metrics.json")))
    diagnostics_file = project_path(str(gen_cfg.get("diagnostics_file", "outputs/reports/generator_fixed_train_diagnostics.json")))
    log_file = project_path(str(gen_cfg.get("log_file", "outputs/logs/generator_fixed_train.log")))
    ensure_dir(output_dir)
    ensure_dir(metrics_file.parent)
    ensure_dir(log_file.parent)

    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)
    max_train = int(gen_cfg.get("max_train_examples", config.get("max_generator_train_examples", 0)) or 0)
    max_val = int(gen_cfg.get("max_val_examples", config.get("max_generator_val_examples", 0)) or 0)
    if max_train:
        train_rows = train_rows[:max_train]
    if max_val:
        val_rows = val_rows[:max_val]
    if not train_rows:
        metrics = {"skipped": True, "reason": f"no generator train rows found at {train_file}"}
        write_json(metrics_file, metrics)
        return metrics

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    local_only = bool(gen_cfg.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_only)
    max_source_length = int(gen_cfg.get("max_source_length", gen_cfg.get("max_input_length", 512)))
    max_target_length = int(gen_cfg.get("max_target_length", 96))
    train_dataset = Seq2SeqRows(train_rows, tokenizer, max_source_length, max_target_length)
    val_dataset = Seq2SeqRows(val_rows, tokenizer, max_source_length, max_target_length) if val_rows else None
    label_diag = label_diagnostics(train_dataset)
    if label_diag["all_ignored_rate"] > 0.05:
        write_json(diagnostics_file, {"label_diagnostics": label_diag, "failure": "all_ignored_labels_above_5_percent"})
        raise RuntimeError(f"Generator labels invalid: {label_diag['all_ignored_rate']:.3%} examples have all ignored labels")
    if label_diag["all_ignored_rate"] > 0.01:
        label_warning = f"WARNING: {label_diag['all_ignored_rate']:.3%} examples have all ignored labels"
    else:
        label_warning = ""

    batch_size = int(gen_cfg.get("batch_size", config.get("generator_batch_size", 2)))
    grad_accum = max(1, int(gen_cfg.get("gradient_accumulation_steps", config.get("gradient_accumulation_steps", 8))))
    epochs = int(gen_cfg.get("epochs", config.get("generator_epochs", 1)))
    lr = float(gen_cfg.get("learning_rate", config.get("generator_lr", 5e-5)))
    weight_decay = float(gen_cfg.get("weight_decay", 0.01))
    max_grad_norm = float(gen_cfg.get("gradient_clip_norm", gen_cfg.get("max_grad_norm", 1.0)))
    fp16 = bool(gen_cfg.get("fp16", False)) and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() and str(config.get("device", "auto")) != "cpu" else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    log_steps = max(1, int(gen_cfg.get("logging_steps", 50)))
    zero_loss_steps = 0
    total_steps = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    failure: dict[str, Any] | None = None

    with log_file.open("w", encoding="utf-8") as log:
        log.write(
            f"model={model_name}\ntrain_examples={len(train_rows)}\nval_examples={len(val_rows)}\n"
            f"device={device}\nfp16={fp16}\nlabel_diagnostics={label_diag}\n{label_warning}\n"
        )
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=fp16):
                    loss = model(**batch).loss / grad_accum
                if not torch.isfinite(loss):
                    failure = _failure("non_finite_loss", epoch, step, batch, tokenizer, loss=float(loss.detach().cpu()))
                    break
                scaler.scale(loss).backward()
                if step % grad_accum == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if not torch.isfinite(grad_norm):
                        failure = _failure("non_finite_grad_norm", epoch, step, batch, tokenizer, grad_norm=float(grad_norm.detach().cpu()))
                        break
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    total_steps += 1
                    step_loss = float((loss.detach().cpu() * grad_accum).item())
                    running_loss += step_loss
                    zero_loss_steps += int(abs(step_loss) < 1e-12)
                    if total_steps % log_steps == 0:
                        log.write(f"epoch={epoch + 1} opt_step={total_steps} train_loss={step_loss:.6f} grad_norm={float(grad_norm):.6f}\n")
                if failure:
                    break
            train_loss = running_loss / max(1, total_steps if epoch == 0 else len(train_loader) // grad_accum)
            eval_loss = evaluate_loss(model, val_loader, device, fp16) if val_loader else None
            epoch_metrics = {"epoch": epoch + 1, "train_loss": train_loss, "eval_loss": eval_loss}
            history.append(epoch_metrics)
            log.write(f"epoch={epoch + 1} summary={epoch_metrics}\n")
            if eval_loss is not None and (math.isnan(eval_loss) or math.isinf(eval_loss)):
                failure = {"reason": "non_finite_eval_loss", "epoch": epoch + 1, "eval_loss": eval_loss}
            if zero_loss_steps > max(20, total_steps // 5):
                log.write(f"WARNING: zero_loss_steps={zero_loss_steps} total_optimizer_steps={total_steps}\n")
            if failure:
                break
    diagnostics = {
        "label_diagnostics": label_diag,
        "zero_loss_steps": zero_loss_steps,
        "optimizer_steps": total_steps,
        "failure": failure,
        "history": history,
    }
    write_json(diagnostics_file, diagnostics)
    if failure:
        raise RuntimeError(f"Generator training stopped: {failure}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    final_eval_loss = history[-1].get("eval_loss") if history else None
    metrics = {
        "model_name": model_name,
        "output_dir": str(output_dir.relative_to(project_path())),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "gradient_clip_norm": max_grad_norm,
        "fp16": fp16,
        "device": str(device),
        "train_runtime": time.perf_counter() - start,
        "train_loss": history[-1]["train_loss"] if history else None,
        "eval_loss": final_eval_loss,
        "eval_perplexity": math.exp(final_eval_loss) if final_eval_loss is not None and final_eval_loss < 20 else None,
        "zero_loss_steps": zero_loss_steps,
        "optimizer_steps": total_steps,
        "label_diagnostics": label_diag,
    }
    write_json(metrics_file, metrics)
    return metrics


class Seq2SeqRows(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_source_length: int, max_target_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        source = row.get("prompt") or row.get("input") or ""
        target = row.get("target") or row.get("target_answer") or row.get("reference_answer") or ""
        source_enc = self.tokenizer(source, max_length=self.max_source_length, truncation=True, padding="max_length")
        target_enc = self.tokenizer(text_target=target, max_length=self.max_target_length, truncation=True, padding="max_length")
        labels = [tok if tok != self.tokenizer.pad_token_id else -100 for tok in target_enc["input_ids"]]
        return {
            "input_ids": torch.tensor(source_enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(source_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def label_diagnostics(dataset: Seq2SeqRows) -> dict:
    all_ignored = 0
    non_ignored_counts = []
    for idx in range(len(dataset)):
        labels = dataset[idx]["labels"]
        non_ignored = int((labels != -100).sum().item())
        non_ignored_counts.append(non_ignored)
        all_ignored += int(non_ignored == 0)
    return {
        "examples": len(dataset),
        "all_ignored": all_ignored,
        "all_ignored_rate": all_ignored / max(1, len(dataset)),
        "min_non_ignored_tokens": min(non_ignored_counts) if non_ignored_counts else 0,
        "avg_non_ignored_tokens": sum(non_ignored_counts) / max(1, len(non_ignored_counts)),
        "max_non_ignored_tokens": max(non_ignored_counts) if non_ignored_counts else 0,
    }


def evaluate_loss(model, loader, device: torch.device, fp16: bool) -> float | None:
    if loader is None:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=fp16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                return float("nan")
            losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / max(1, len(losses))


def _failure(reason: str, epoch: int, step: int, batch: dict[str, torch.Tensor], tokenizer, **extra) -> dict[str, Any]:
    labels = batch["labels"][0].detach().cpu().tolist()
    decoded_labels = tokenizer.decode([tok for tok in labels if tok != -100], skip_special_tokens=True)
    decoded_input = tokenizer.decode(batch["input_ids"][0].detach().cpu().tolist(), skip_special_tokens=True)
    return {
        "reason": reason,
        "epoch": epoch + 1,
        "step": step,
        "decoded_input": decoded_input[:1000],
        "decoded_label": decoded_labels,
        **extra,
    }
