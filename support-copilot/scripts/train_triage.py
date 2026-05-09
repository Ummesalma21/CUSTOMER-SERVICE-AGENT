from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.triage.train_triage import train_triage
from src.utils.io import load_config
parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train_full.yaml"); args = parser.parse_args()
print(train_triage(load_config(args.config)))
