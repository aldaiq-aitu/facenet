from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from device_utils import get_device
from evaluation.inference import score_pairs
from evaluation.io import read_pairs_csv
from evaluation.metrics import compute_verification_metrics, find_best_threshold
from model_registry import build_model, get_model_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained FaceNet as a fixed baseline.")
    parser.add_argument("--model", choices=["facenet"], default="facenet")
    parser.add_argument("--val-pairs-csv", type=Path, required=True)
    parser.add_argument("--test-pairs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("baselines/facenet_pretrained"))
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_model_spec(args.model)
    device = get_device()
    print(f"Using device: {device}")

    model = build_model(args.model).eval().to(device)
    val_pairs = read_pairs_csv(args.val_pairs_csv)
    test_pairs = read_pairs_csv(args.test_pairs_csv)

    val_scores, val_labels = score_pairs(model, val_pairs, spec.input_size, device)
    val_metrics = find_best_threshold(val_scores, val_labels)
    test_scores, test_labels = score_pairs(model, test_pairs, spec.input_size, device)
    test_metrics = compute_verification_metrics(test_scores, test_labels, val_metrics.threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "baseline.pt"
    torch.save(
        {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "input_size": spec.input_size,
            "embedding_size": spec.embedding_size,
            "validation_metrics": val_metrics.to_dict(),
            "baseline_type": "pretrained",
        },
        checkpoint_path,
    )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(val_metrics.to_dict(), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics.to_dict(), indent=2),
        encoding="utf-8",
    )

    print("Validation:", json.dumps(val_metrics.to_dict(), indent=2))
    print("Test:", json.dumps(test_metrics.to_dict(), indent=2))
    print(f"Baseline checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
