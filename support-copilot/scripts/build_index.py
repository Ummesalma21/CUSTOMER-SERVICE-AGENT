from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.build_faiss import build_faiss_index
parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train_full.yaml"); parser.parse_args()
idx = build_faiss_index(); print({"chunks": len(idx.get("chunks", [])), "backend": idx.get("backend")})
