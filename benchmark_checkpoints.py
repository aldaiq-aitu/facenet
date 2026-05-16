from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from model_registry import build_model, get_model_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark trained checkpoints with the same protocol.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-csv", type=Path, default=Path("results/benchmark_checkpoints.csv"))
    return parser.parse_args()


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def model_size_mb(model):
    return sum(parameter.numel() * 4 for parameter in model.parameters()) / (1024 * 1024)


def benchmark_checkpoint(checkpoint_path: Path, runs: int, warmup: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint["model_name"]
    spec = get_model_spec(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    dummy = torch.randn(1, 3, spec.input_size, spec.input_size, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            model(dummy)
    synchronize_if_needed(device)

    latencies = []
    for _ in range(runs):
        synchronize_if_needed(device)
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        synchronize_if_needed(device)
        latencies.append((time.perf_counter() - start) * 1000)

    process = psutil.Process(os.getpid())
    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "params_m": count_params(model),
        "size_mb": model_size_mb(model),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "fps": float(1000.0 / np.mean(latencies)),
        "rss_mb": process.memory_info().rss / (1024 * 1024),
        "device": str(device),
    }


def main():
    args = parse_args()
    rows = [benchmark_checkpoint(path, args.runs, args.warmup) for path in args.checkpoint]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Saved benchmark results to {args.output_csv}")


if __name__ == "__main__":
    main()
