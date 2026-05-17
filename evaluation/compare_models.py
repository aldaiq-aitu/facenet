from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from device_utils import get_device
from evaluation.inference import score_pairs
from evaluation.io import read_pairs_csv
from evaluation.metrics import compute_verification_metrics
from model_registry import build_model, get_model_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints on the same test pairs.")
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("results/model_comparison.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    pairs = read_pairs_csv(args.pairs_csv)
    rows = []

    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_name = checkpoint["model_name"]
        spec = get_model_spec(model_name)
        model = build_model(model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(device)

        scores, labels = score_pairs(model, pairs, spec.input_size, device)
        validation_threshold = checkpoint["validation_metrics"]["threshold"]
        metrics = compute_verification_metrics(scores, labels, validation_threshold)
        rows.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "baseline_type": checkpoint.get("baseline_type", "trained"),
                **metrics.to_dict(),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))
    print(f"Saved comparison to {args.output_csv}")


if __name__ == "__main__":
    main()
