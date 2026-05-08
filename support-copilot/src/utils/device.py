from __future__ import annotations


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

