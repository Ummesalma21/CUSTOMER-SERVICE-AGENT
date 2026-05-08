from __future__ import annotations

import math


def boundary_loss(logits: list[float], gold_idx: int, mu: float = 0.15) -> float:
    correct = logits[gold_idx]
    wrong = max(v for i, v in enumerate(logits) if i != gold_idx)
    return math.log1p(math.exp(wrong - correct + mu))


def margin(logits: list[float], pred_idx: int) -> float:
    pred = logits[pred_idx]
    wrong = max(v for i, v in enumerate(logits) if i != pred_idx)
    return pred - wrong

