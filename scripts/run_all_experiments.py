from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    Path("configs/facenet.yaml"),
    Path("configs/mobilefacenet.yaml"),
    Path("configs/efficientnet_lite0.yaml"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train all models using the configured protocol.")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    for config in CONFIGS:
        print(f"\n=== Training from {config} ===")
        subprocess.run(
            [args.python, "-m", "training.train", "--config", str(config)],
            check=True,
        )


if __name__ == "__main__":
    main()
