from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.train_generator_lora import train_generator_lora
from src.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator_fixed.yaml")
    args = parser.parse_args()
    print(train_generator_lora(load_config(args.config)))


if __name__ == "__main__":
    main()
