# Model Usage Guide

The repository compares three backbones under one face-recognition protocol.

| Config name | Input | Embedding | Intended role |
|---|---:|---:|---|
| `facenet` | 160x160 | 512-d | strong baseline |
| `mobilefacenet` | 112x112 | 128-d | lightweight mobile candidate |
| `efficientnet_lite0` | 112x112 | 512-d | efficient CNN candidate |

## Important methodological note

All three backbones should be evaluated only after the shared research pipeline:

1. aligned inputs;
2. ArcFace training/fine-tuning;
3. validation threshold calibration;
4. held-out test evaluation.

`FaceNet` starts from pretrained face-recognition weights.

`MobileFaceNet` is implemented as a compact MobileFaceNet-style backbone and
must be trained on face identities.

`EfficientNet-Lite0` starts from an ImageNet-pretrained backbone and must be
fine-tuned for face recognition before being compared with the other models.

## Switching the runtime backbone

Edit `config.py`:

```python
BACKBONE = "mobilefacenet"
```

Available values:

```python
"facenet"
"mobilefacenet"
"efficientnet_lite0"
```

## Training

Use the predefined experiment configs:

```bash
python -m training.train --config configs/facenet.yaml
python -m training.train --config configs/mobilefacenet.yaml
python -m training.train --config configs/efficientnet_lite0.yaml
```

Or train all three:

```bash
python -m scripts.run_all_experiments
```

## Database compatibility

Embeddings produced by different backbones are not interchangeable. If the
runtime backbone changes, enroll identities again before using the webcam demo.
