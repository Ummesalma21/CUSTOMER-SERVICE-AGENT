from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.train_retriever import train_retriever
from src.utils.io import load_config
parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/smoke.yaml"); args = parser.parse_args()
print(train_retriever(load_config(args.config)))

