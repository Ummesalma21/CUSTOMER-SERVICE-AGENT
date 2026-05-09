from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.grounded_generator import generate_grounded_answer
from src.generation.train_generator_lora import train_generator_lora
from src.utils.io import load_config, project_path, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator_overfit.yaml")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = train_generator_lora(config)
    gen_cfg = config["generator_training"]
    rows = read_jsonl(project_path("data", "processed", "generator_train.jsonl"))[: int(gen_cfg.get("max_train_examples", 96))]
    predictions = []
    for row in rows[: args.examples]:
        result = generate_grounded_answer(
            query=row["query"],
            evidence_passages=row.get("evidence", [])[:3],
            model_name=str(gen_cfg.get("output_dir", "outputs/generator/flan_t5_overfit_check")),
            fallback_model_name=str(gen_cfg.get("output_dir", "outputs/generator/flan_t5_overfit_check")),
            max_new_tokens=96,
            num_beams=2,
            do_sample=False,
        )
        predictions.append(
            {
                "query": row.get("query"),
                "reference": row.get("target"),
                "prediction": result.get("answer"),
                "status": result.get("status"),
                "model_name": result.get("model_name"),
            }
        )
    write_jsonl(project_path("outputs", "reports", "generator_overfit_predictions.jsonl"), predictions)
    project_path("outputs", "reports", "generator_overfit_check.md").write_text(_summary(metrics, predictions), encoding="utf-8")
    print({"metrics": metrics, "predictions": predictions[:2]})


def _summary(metrics: dict, predictions: list[dict]) -> str:
    losses = [item.get("train_loss") for item in metrics.get("history", []) if item.get("train_loss") is not None]
    lines = [
        "# Generator Overfit Check",
        "",
        "This is a small debugging run only; it does not replace the full generator dataset.",
        "",
        f"Train examples: `{metrics.get('train_examples')}`",
        f"Validation examples: `{metrics.get('val_examples')}`",
        f"Final train loss: `{metrics.get('train_loss')}`",
        f"Final eval loss: `{metrics.get('eval_loss')}`",
        f"Finite losses: `{metrics.get('train_loss') is not None and metrics.get('eval_loss') is not None}`",
        "",
        "## Sample Predictions",
        "",
    ]
    for pred in predictions:
        lines.extend(
            [
                f"- Query: `{pred.get('query')}`",
                f"  Reference: `{pred.get('reference')}`",
                f"  Prediction: `{pred.get('prediction')}`",
                f"  Status/model: `{pred.get('status')}` / `{pred.get('model_name')}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
