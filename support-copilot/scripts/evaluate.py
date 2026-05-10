from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_all import evaluate_all
from src.utils.io import load_config
parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/proposed.yaml"); args = parser.parse_args()
print(evaluate_all(load_config(args.config)))
