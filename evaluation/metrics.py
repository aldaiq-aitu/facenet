from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerificationMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float


def compute_verification_metrics(scores, labels, threshold: float) -> VerificationMetrics:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    predictions = scores >= threshold

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    total = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / max(1e-12, precision + recall)) if (precision + recall) else 0.0
    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)
    accuracy = (tp + tn) / total

    return VerificationMetrics(threshold, accuracy, precision, recall, f1, far, frr)


def find_best_threshold(scores, labels, num_steps: int = 1001) -> VerificationMetrics:
    candidates = np.linspace(-1.0, 1.0, num_steps)
    results = [compute_verification_metrics(scores, labels, float(th)) for th in candidates]
    return max(results, key=lambda item: (item.accuracy, item.f1))
