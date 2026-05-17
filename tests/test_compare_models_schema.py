from evaluation.metrics import VerificationMetrics


def test_verification_metrics_dict_contains_baseline_fields():
    metrics = VerificationMetrics(
        threshold=0.3,
        accuracy=0.9,
        precision=0.9,
        recall=0.9,
        f1=0.9,
        far=0.1,
        frr=0.1,
        roc_auc=0.95,
        eer=0.1,
    )
    payload = metrics.to_dict()
    assert "accuracy" in payload
    assert "eer" in payload
