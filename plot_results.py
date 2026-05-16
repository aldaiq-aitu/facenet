from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create thesis-ready comparison plots.")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def save_bar(df, x, y, title, ylabel, path):
    plt.figure(figsize=(7, 5))
    plt.bar(df[x], df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    metrics = pd.read_csv(args.metrics_csv)
    benchmark = pd.read_csv(args.benchmark_csv)
    merged = metrics.merge(benchmark, on="model")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_bar(merged, "model", "accuracy", "Verification Accuracy", "Accuracy", args.output_dir / "accuracy.png")
    save_bar(merged, "model", "eer", "Equal Error Rate", "EER", args.output_dir / "eer.png")
    save_bar(merged, "model", "latency_mean_ms", "Inference Latency", "Milliseconds", args.output_dir / "latency.png")
    save_bar(merged, "model", "size_mb", "Model Size", "MB", args.output_dir / "model_size.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(merged["latency_mean_ms"], merged["accuracy"])
    for _, row in merged.iterrows():
        plt.annotate(row["model"], (row["latency_mean_ms"], row["accuracy"]))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency")
    plt.tight_layout()
    plt.savefig(args.output_dir / "accuracy_vs_latency.png", dpi=200)
    plt.close()

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
