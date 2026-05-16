from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_BY_MODEL = {
    "facenet": Path("configs/facenet.yaml"),
    "mobilefacenet": Path("configs/mobilefacenet.yaml"),
    "efficientnet_lite0": Path("configs/efficientnet_lite0.yaml"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one assigned model on one device.")
    parser.add_argument("--model", choices=sorted(CONFIG_BY_MODEL), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = CONFIG_BY_MODEL[args.model]
    experiment_dir = Path("experiments") / args.model
    checkpoint = experiment_dir / "best.pt"
    test_pairs = Path("data/pairs/test_pairs.csv")
    results_dir = Path("results") / args.model

    subprocess.run([args.python, "-m", "training.train", "--config", str(config)], check=True)
    subprocess.run(
        [
            args.python,
            "-m",
            "evaluation.evaluate_pairs",
            "--checkpoint",
            str(checkpoint),
            "--pairs-csv",
            str(test_pairs),
            "--threshold",
            _read_validation_threshold(checkpoint),
            "--output-dir",
            str(results_dir),
        ],
        check=True,
    )

    if not args.skip_benchmark:
        subprocess.run(
            [
                args.python,
                "benchmark_checkpoints.py",
                "--checkpoint",
                str(checkpoint),
                "--output-csv",
                str(results_dir / "benchmark.csv"),
            ],
            check=True,
        )


def _read_validation_threshold(checkpoint_path: Path) -> str:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return str(checkpoint["validation_metrics"]["threshold"])


if __name__ == "__main__":
    main()
