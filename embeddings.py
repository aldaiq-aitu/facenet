"""
Model loading, embedding extraction, and nearest-neighbor recognition.

All backbones are created through the shared registry so inference, training,
and evaluation use the same model metadata and preprocessing contract.
"""

import logging

import numpy as np
import torch
from facenet_pytorch import MTCNN

from config import BACKBONE, CHECKPOINT_PATH, MIN_FACE_SIZE, THRESHOLD, USE_CHECKPOINT_THRESHOLD
from model_registry import build_model, get_model_spec
from preprocessing import prepare_face_tensor

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

mtcnn = MTCNN(keep_all=True, device=device, min_face_size=MIN_FACE_SIZE)

spec = get_model_spec(BACKBONE)
model = build_model(BACKBONE).eval().to(device)
INPUT_SIZE = spec.input_size
runtime_threshold = THRESHOLD

if CHECKPOINT_PATH:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if checkpoint["model_name"] != BACKBONE:
        raise ValueError(
            f"Checkpoint model '{checkpoint['model_name']}' does not match BACKBONE '{BACKBONE}'."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    if USE_CHECKPOINT_THRESHOLD:
        runtime_threshold = float(checkpoint["validation_metrics"]["threshold"])
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

print(f"Backbone: {spec.name} ({spec.embedding_size}-d, {spec.input_size}x{spec.input_size})")


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def get_embedding_from_crop(face_crop_pil):
    """Extract an L2-normalized embedding from a PIL face crop."""
    try:
        tensor = prepare_face_tensor(face_crop_pil.resize((INPUT_SIZE, INPUT_SIZE)))
        with torch.no_grad():
            embedding = model(tensor.unsqueeze(0).to(device))
        return normalize(embedding[0].cpu().numpy())
    except Exception:
        logger.exception("Embedding extraction failed")
        return None


def recognize(embedding, database):
    """Compare an embedding against all stored vectors using cosine similarity."""
    best_name, best_score = "Unknown", 0.0
    query = normalize(np.asarray(embedding).flatten())

    for name, vectors in database.items():
        if name.startswith("__"):
            continue
        for vector in vectors:
            stored = normalize(np.asarray(vector).flatten())
            score = float(np.dot(query, stored))
            if score > best_score:
                best_score, best_name = score, name

    if best_score > runtime_threshold:
        return best_name, best_score
    return "Unknown", best_score
