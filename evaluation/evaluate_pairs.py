from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from evaluation.inference import score_pairs
from evaluation.io import read_pairs_csv
from evaluation.metrics import compute_verification_metrics, find_best_threshold, roc_curve
from model_registry import build_model, get_model_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on verification pairs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def save_outputs(output_dir: Path, metrics, scores, labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")

    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["score", "label"])
        writer.writerows(zip(scores, labels))

    fprs, tprs, _ = roc_curve(scores, labels)
    plt.figure(figsize=(6, 5))
    plt.plot(fprs, tprs, label=f"AUC = {metrics.roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]
    spec = get_model_spec(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    pairs = read_pairs_csv(args.pairs_csv)
    scores, labels = score_pairs(model, pairs, spec.input_size, device)

    metrics = (
        compute_verification_metrics(scores, labels, args.threshold)
        if args.threshold is not None
        else find_best_threshold(scores, labels)
    )
    print(
        "threshold={:.4f} accuracy={:.4f} precision={:.4f} recall={:.4f} "
        "f1={:.4f} FAR={:.4f} FRR={:.4f} AUC={:.4f} EER={:.4f}".format(
            metrics.threshold,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.far,
            metrics.frr,
            metrics.roc_auc,
            metrics.eer,
        )
    )

    if args.output_dir is not None:
        save_outputs(args.output_dir, metrics, scores, labels)


if __name__ == "__main__":
    main()
