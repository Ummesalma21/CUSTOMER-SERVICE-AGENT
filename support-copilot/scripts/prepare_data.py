from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import prepare_data
from src.utils.io import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train_full.yaml")
args = parser.parse_args()
prepare_data(load_config(args.config))
