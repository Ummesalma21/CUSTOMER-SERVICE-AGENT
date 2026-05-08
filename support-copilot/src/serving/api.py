from __future__ import annotations

from src.evaluation.evaluate_end_to_end import run_proposed
from src.utils.io import load_config

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover
    FastAPI = None

if FastAPI:
    app = FastAPI(title="Reject-Aware Support Copilot")
    CONFIG = load_config("configs/smoke.yaml")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/answer")
    def answer(payload: dict) -> dict:
        return run_proposed(payload.get("query", ""), CONFIG)
else:
    app = None

