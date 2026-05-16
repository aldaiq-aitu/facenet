from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.metrics import find_best_threshold
from model_registry import build_model, get_model_spec
from preprocessing import prepare_face_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on verification pairs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(a, b))


def load_embedding(model, image_path: Path, input_size: int, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((input_size, input_size))
    tensor = prepare_face_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)[0].cpu().numpy()
    return embedding


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]
    spec = get_model_spec(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    scores, labels = [], []
    with args.pairs_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"path1", "path2", "is_same"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"pairs CSV must contain exactly columns: {sorted(required)}")

        for row in reader:
            emb1 = load_embedding(model, Path(row["path1"]), spec.input_size, device)
            emb2 = load_embedding(model, Path(row["path2"]), spec.input_size, device)
            scores.append(cosine_similarity(emb1, emb2))
            labels.append(int(row["is_same"]))

    best = find_best_threshold(scores, labels)
    print(
        "threshold={:.4f} accuracy={:.4f} precision={:.4f} recall={:.4f} "
        "f1={:.4f} FAR={:.4f} FRR={:.4f}".format(
            best.threshold,
            best.accuracy,
            best.precision,
            best.recall,
            best.f1,
            best.far,
            best.frr,
        )
    )


if __name__ == "__main__":
    main()
