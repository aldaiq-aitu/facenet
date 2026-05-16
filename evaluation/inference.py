from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from preprocessing import prepare_face_tensor


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


def score_pairs(model, pairs, input_size: int, device: torch.device):
    cache = {}
    scores, labels = [], []
    model.eval()
    for path1, path2, label in pairs:
        if path1 not in cache:
            cache[path1] = load_embedding(model, path1, input_size, device)
        if path2 not in cache:
            cache[path2] = load_embedding(model, path2, input_size, device)
        scores.append(cosine_similarity(cache[path1], cache[path2]))
        labels.append(label)
    return scores, labels
