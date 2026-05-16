import pytest

from evaluation.metrics import compute_verification_metrics, find_best_threshold


def test_find_best_threshold_perfect_separation():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1, 1, 0, 0]
    metrics = find_best_threshold(scores, labels)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)


def test_compute_metrics_basic_values():
    metrics = compute_verification_metrics([0.9, 0.8, 0.7, 0.1], [1, 1, 0, 0], threshold=0.75)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.far == pytest.approx(0.0)
    assert metrics.frr == pytest.approx(0.0)
