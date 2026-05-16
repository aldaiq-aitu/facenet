"""
benchmark.py — Compare face recognition models for energy efficiency.

Models tested:
    1. FaceNet                       — baseline (heavy)
    2. MobileFaceNet                 — lightweight
    3. EfficientNet-Lite0            — balanced accuracy/speed

Metrics collected:
    - Number of parameters (M)
    - Model size on disk (MB)
    - Inference latency (ms)
    - Theoretical FPS (model-only)
    - RAM usage (MB)
    - CPU load (%)
    - Energy-Efficiency Score (params × latency)

Usage:
    python benchmark.py
"""

import torch
import time
import numpy as np
import psutil
import os

from models.facenet import FaceNet
from models.mobilefacenet import MobileFaceNet
from models.efficientnet_lite import EfficientNetLite0Face

# ─── Helpers ────────────────────────────────────────────────────────────────

def count_params(model):
    """Total trainable parameters in Millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(model):
    """Approximate model size in MB (float32 weights only)."""
    return sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)


# ─── Benchmark runner ───────────────────────────────────────────────────────

def benchmark(name, model, input_shape, num_runs=200):
    """
    Run `num_runs` forward passes and collect performance stats.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
    dummy = torch.randn(*input_shape).to(device)

    params   = count_params(model)
    size_mb  = model_size_mb(model)

    # Warm-up
    for _ in range(20):
        with torch.no_grad():
            model(dummy)

    # Measure latency
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps     = 1000.0 / avg_lat

    # System resources
    process = psutil.Process(os.getpid())
    mem_mb  = process.memory_info().rss / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=0.5)

    return {
        'name':       name,
        'params_m':   params,
        'size_mb':    size_mb,
        'avg_ms':     avg_lat,
        'std_ms':     std_lat,
        'min_ms':     min_lat,
        'max_ms':     max_lat,
        'fps':        fps,
        'mem_mb':     mem_mb,
        'cpu_pct':    cpu_pct,
    }


def print_result(r):
    """Pretty-print a single benchmark result."""
    print(f"\n{'=' * 50}")
    print(f"  Model:  {r['name']}")
    print(f"{'=' * 50}")
    print(f"  Parameters:        {r['params_m']:.2f} M")
    print(f"  Model size:        {r['size_mb']:.2f} MB")
    print(f"  Avg latency:       {r['avg_ms']:.2f} ms  (±{r['std_ms']:.2f})")
    print(f"  Min / Max:         {r['min_ms']:.2f} / {r['max_ms']:.2f} ms")
    print(f"  Throughput (FPS):  {r['fps']:.1f}")
    print(f"  Memory (RSS):      {r['mem_mb']:.1f} MB")
    print(f"  CPU load:          {r['cpu_pct']:.1f} %")


def print_comparison(results):
    """Side-by-side comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON TABLE  —  3 Models")
    print(f"{'=' * 80}")

    # Dynamic column width
    col_w = 18
    header = f"{'Metric':<20}"
    for r in results:
        header += f"  {r['name']:>{col_w}}"
    print(header)
    print("-" * 80)

    rows = [
        ('Parameters (M)',  'params_m',  '{:.2f}'),
        ('Model size (MB)', 'size_mb',   '{:.2f}'),
        ('Latency (ms)',    'avg_ms',    '{:.2f}'),
        ('FPS',             'fps',       '{:.1f}'),
        ('Memory (MB)',     'mem_mb',    '{:.1f}'),
        ('CPU (%)',         'cpu_pct',   '{:.1f}'),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<18}"
        for r in results:
            line += f"  {fmt.format(r[key]):>{col_w}}"
        print(line)

    print("-" * 80)

    # Energy-efficiency score
    print(f"\n  Energy-Efficiency Score  (params × latency, lower = better):")
    scores = [(r['name'], r['params_m'] * r['avg_ms']) for r in results]
    scores.sort(key=lambda x: x[1])
    for i, (name, score) in enumerate(scores):
        medal = ["🥇", "🥈", "🥉", "  "][i]
        print(f"    {medal} {name:<25} → {score:.2f}")


def save_results(results, path="benchmark_results.txt", num_runs=200):
    """Persist results to a text file."""
    with open(path, 'w') as f:
        f.write("BENCHMARK RESULTS — Face Recognition Models\n")
        f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Runs per model: {num_runs}\n")
        f.write(f"{'=' * 60}\n\n")

        for r in results:
            f.write(f"Model: {r['name']}\n")
            f.write(f"  Parameters:   {r['params_m']:.2f} M\n")
            f.write(f"  Model size:   {r['size_mb']:.2f} MB\n")
            f.write(f"  Latency:      {r['avg_ms']:.2f} ms (±{r['std_ms']:.2f})\n")
            f.write(f"  FPS:          {r['fps']:.1f}\n")
            f.write(f"  Memory:       {r['mem_mb']:.1f} MB\n")
            f.write(f"  CPU:          {r['cpu_pct']:.1f} %\n\n")

        # Comparison summary
        f.write(f"{'=' * 60}\n")
        f.write("Energy-Efficiency Score (params × latency):\n")
        scores = [(r['name'], r['params_m'] * r['avg_ms']) for r in results]
        scores.sort(key=lambda x: x[1])
        for name, score in scores:
            f.write(f"  {name:<25} → {score:.2f}\n")

    print(f"\nResults saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = 'CUDA' if torch.cuda.is_available() else 'CPU'
    print(f"Running benchmarks on: {device}")
    print("This may take 1-2 minutes...\n")

    results = []

    # ── 1. FaceNet (Baseline) ─────────────────────────────────────────
    print("[1/3] Benchmarking FaceNet...")
    m1 = FaceNet()
    r1 = benchmark("FaceNet", m1, (1, 3, 160, 160))
    print_result(r1)
    results.append(r1)
    del m1

    # ── 2. MobileFaceNet ──────────────────────────────────────────────
    print("\n[2/3] Benchmarking MobileFaceNet...")
    m2 = MobileFaceNet(embedding_size=128, input_size=112)
    r2 = benchmark("MobileFaceNet", m2, (1, 3, 112, 112))
    print_result(r2)
    results.append(r2)
    del m2

    # ── 3. EfficientNet-Lite0 ─────────────────────────────────────────
    print("\n[3/3] Benchmarking EfficientNet-Lite0...")
    m3 = EfficientNetLite0Face(embedding_size=512, pretrained=True)
    r3 = benchmark("EfficientNet-Lite0", m3, (1, 3, 112, 112))
    print_result(r3)
    results.append(r3)
    del m3

    # ── Comparison ────────────────────────────────────────────────────
    print_comparison(results)
    save_results(results)
