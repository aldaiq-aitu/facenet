from __future__ import annotations

from dataclasses import asdict, dataclass

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
    roc_auc: float
    eer: float

    def to_dict(self):
        return asdict(self)


def _confusion(scores, labels, threshold):
    predictions = scores >= threshold
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    return tp, tn, fp, fn


def roc_curve(scores, labels):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    thresholds = np.concatenate(([np.inf], np.sort(np.unique(scores))[::-1], [-np.inf]))
    tprs, fprs = [], []
    for threshold in thresholds:
        tp, tn, fp, fn = _confusion(scores, labels, float(threshold))
        tprs.append(tp / max(1, tp + fn))
        fprs.append(fp / max(1, fp + tn))
    return np.asarray(fprs), np.asarray(tprs), thresholds


def roc_auc_score(scores, labels) -> float:
    fprs, tprs, _ = roc_curve(scores, labels)
    order = np.argsort(fprs)
    x = fprs[order]
    y = tprs[order]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def equal_error_rate(scores, labels) -> float:
    fprs, tprs, _ = roc_curve(scores, labels)
    fnrs = 1.0 - tprs
    idx = int(np.argmin(np.abs(fprs - fnrs)))
    return float((fprs[idx] + fnrs[idx]) / 2.0)


def compute_verification_metrics(scores, labels, threshold: float) -> VerificationMetrics:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    tp, tn, fp, fn = _confusion(scores, labels, threshold)

    total = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / max(1e-12, precision + recall)) if (precision + recall) else 0.0
    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)
    accuracy = (tp + tn) / total

    return VerificationMetrics(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        far=far,
        frr=frr,
        roc_auc=roc_auc_score(scores, labels),
        eer=equal_error_rate(scores, labels),
    )


def find_best_threshold(scores, labels, num_steps: int = 1001) -> VerificationMetrics:
    candidates = np.linspace(-1.0, 1.0, num_steps)
    results = [compute_verification_metrics(scores, labels, float(th)) for th in candidates]
    return max(results, key=lambda item: (item.accuracy, item.f1, -item.eer))
