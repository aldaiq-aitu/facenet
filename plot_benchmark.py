import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

BENCHMARK_RESULTS_FILE = Path("benchmark_results.txt")


def parse_benchmark_results(path=BENCHMARK_RESULTS_FILE):
    """Read benchmark.py output and return per-model chart data."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"Model: (?P<name>.+?)\n"
        r"\s+Parameters:\s+(?P<params>[\d.]+) M\n"
        r"\s+Model size:\s+(?P<size>[\d.]+) MB\n"
        r"\s+Latency:\s+(?P<latency>[\d.]+) ms.*?\n"
        r"\s+FPS:\s+(?P<fps>[\d.]+)",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(text):
        params = float(match.group("params"))
        latency = float(match.group("latency"))
        rows.append(
            {
                "model": match.group("name"),
                "params": params,
                "latency": latency,
                "fps": float(match.group("fps")),
                "size_mb": float(match.group("size")),
                "eff_score": params * latency,
            }
        )

    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")

    return rows


def annotate_bars(bars, suffix="", offset=0.5):
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + offset,
            f"{yval:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def main():
    rows = parse_benchmark_results()
    models = [row["model"] for row in rows]
    params = [row["params"] for row in rows]
    latency = [row["latency"] for row in rows]
    fps = [row["fps"] for row in rows]
    size_mb = [row["size_mb"] for row in rows]
    eff_score = [row["eff_score"] for row in rows]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, params, color=colors)
    plt.title("Модель параметры (Миллионы)", fontsize=14, pad=15)
    plt.ylabel("Параметры (M)")
    annotate_bars(bars, "M")
    plt.tight_layout()
    plt.savefig("graph_parameters.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, fps, color=colors)
    plt.title("Пропускная способность (FPS)", fontsize=14, pad=15)
    plt.ylabel("FPS")
    annotate_bars(bars, offset=1)
    plt.tight_layout()
    plt.savefig("graph_fps.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, eff_score, color=colors)
    plt.title("Оценка энергоэффективности (Меньше = Лучше)", fontsize=14, pad=15)
    plt.ylabel("Score (Params x Latency)")
    annotate_bars(bars, offset=10)
    plt.tight_layout()
    plt.savefig("graph_efficiency.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(models, latency, marker="o", linestyle="-", linewidth=2, markersize=8, color="#9b59b6", label="Задержка (ms)")
    plt.plot(models, size_mb, marker="s", linestyle="--", linewidth=2, markersize=8, color="#f39c12", label="Размер (MB)")
    plt.title("Размер vs Задержка", fontsize=14, pad=15)
    plt.ylabel("MB / ms")
    plt.legend()
    for i in range(len(models)):
        plt.text(i, latency[i] + 5, f"{latency[i]:.2f}ms", ha="center", color="#9b59b6", fontweight="bold")
        plt.text(i, size_mb[i] - 10, f"{size_mb[i]:.2f}MB", ha="center", color="#f39c12", fontweight="bold")
    plt.tight_layout()
    plt.savefig("graph_line_tradeoff.png", dpi=300)
    plt.close()

    print(f"Graphs generated successfully from {BENCHMARK_RESULTS_FILE}.")


if __name__ == "__main__":
    main()
