from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preference.train_preference_ranker import train_preference_ranker
from src.utils.io import load_config
parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/smoke.yaml"); args = parser.parse_args()
print(train_preference_ranker(load_config(args.config)))

