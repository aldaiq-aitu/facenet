from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image


REFERENCE_FIVE_POINTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class FaceCrop:
    image: Image.Image
    aligned: bool


def _scaled_reference_points(output_size: int) -> np.ndarray:
    return REFERENCE_FIVE_POINTS_112 * (output_size / 112.0)


def align_face(
    frame_rgb: np.ndarray,
    landmarks: Iterable[Iterable[float]] | None,
    output_size: int,
) -> FaceCrop | None:
    """Align a face using five landmarks; return None when alignment is impossible."""
    if landmarks is None:
        return None

    src = np.asarray(landmarks, dtype=np.float32)
    if src.shape != (5, 2):
        return None

    dst = _scaled_reference_points(output_size)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None

    warped = cv2.warpAffine(
        frame_rgb,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return FaceCrop(image=Image.fromarray(warped), aligned=True)


def crop_face(frame_rgb: np.ndarray, box: Iterable[float], output_size: int) -> FaceCrop | None:
    """Fallback crop when landmarks are unavailable."""
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None

    image = Image.fromarray(frame_rgb[y1:y2, x1:x2]).resize((output_size, output_size))
    return FaceCrop(image=image, aligned=False)


def prepare_face_tensor(face_image: Image.Image) -> torch.Tensor:
    """Convert a face crop to the normalized tensor used by all backbones."""
    array = np.asarray(face_image).astype(np.float32)
    array = (array / 255.0 - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1)
