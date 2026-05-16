# Diploma Workflow

## Research question

Which of the three face-recognition backbones provides the best balance between
verification quality and computational efficiency under a shared protocol?

## Compared models

| Model | Role |
|---|---|
| FaceNet | accuracy-oriented baseline |
| MobileFaceNet | lightweight mobile candidate |
| EfficientNet-Lite0 | efficient CNN candidate |

## Fixed protocol

All models must use:

1. the same aligned images;
2. the same identity-disjoint split policy;
3. the same ArcFace training objective;
4. the same validation and held-out test pair protocol;
5. model-specific thresholds calibrated on validation only;
6. the same benchmark machine and benchmark script.

## Final experimental artifacts

| Artifact | Purpose |
|---|---|
| `alignment_manifest.csv` | data preparation audit trail |
| `split_manifest.csv` | proof of train/val/test composition |
| `history.csv` | training curves |
| `best.pt` | selected checkpoint |
| `model_comparison.csv` | quality metrics on held-out test pairs |
| `benchmark_checkpoints.csv` | efficiency metrics |
| `plots/*.png` | thesis figures |

## Metrics to report

### Quality

- accuracy
- precision
- recall
- F1-score
- FAR
- FRR
- ROC-AUC
- EER

### Efficiency

- parameter count
- model size
- mean latency
- p95 latency
- FPS
- RSS memory

## Report structure

1. introduction and motivation;
2. related work;
3. system design;
4. dataset and preprocessing;
5. training protocol;
6. experimental setup;
7. results and discussion;
8. conclusion and future work.

## What counts as a valid conclusion

The final claim must be based on both quality and efficiency. A smaller model is
not automatically better if its verification quality drops too far, and a more
accurate model is not automatically preferable if it violates the deployment
constraints of the intended system.
