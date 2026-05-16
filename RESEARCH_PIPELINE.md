# Research Pipeline

This repository now separates the real-time demo from the research workflow used
to compare the three face-recognition backbones fairly.

## Goal

Train and evaluate all models under the same protocol:

1. aligned face crops;
2. shared normalization;
3. ArcFace training head;
4. cosine-similarity verification during evaluation;
5. per-model threshold calibration;
6. common efficiency benchmarks.

## Current model set

| Model | Input | Embedding | Role |
|---|---:|---:|---|
| `facenet` | 160x160 | 512 | high-accuracy baseline |
| `mobilefacenet` | 112x112 | 128 | lightweight mobile model |
| `efficientnet_lite0` | 112x112 | 512 | efficient CNN candidate |

## Expected dataset layout for training

Use aligned face images arranged like `torchvision.datasets.ImageFolder`:

```text
data/train/
  person_001/
    image_001.jpg
    image_002.jpg
  person_002/
    image_001.jpg
```

The current `training/train.py` script assumes that the images are already
aligned. For raw face datasets, run an alignment/preprocessing step before
training so every model receives equivalent inputs.

## Train

```bash
python -m training.train --model facenet --train-dir data/train
python -m training.train --model mobilefacenet --train-dir data/train
python -m training.train --model efficientnet_lite0 --train-dir data/train
```

## Evaluate verification quality

Prepare a CSV with exactly three columns:

```csv
path1,path2,is_same
data/val/a_1.jpg,data/val/a_2.jpg,1
data/val/a_1.jpg,data/val/b_1.jpg,0
```

Then run:

```bash
python -m evaluation.evaluate_pairs --checkpoint checkpoints/mobilefacenet_arcface.pt --pairs-csv data/pairs_val.csv
```

The evaluator searches for the best cosine-similarity threshold and reports:

- accuracy;
- precision;
- recall;
- F1;
- FAR;
- FRR.

## Still required before final diploma experiments

1. choose and prepare the training dataset;
2. add a deterministic alignment stage for the full dataset;
3. introduce train/validation/test splits;
4. add learning-rate scheduling, validation, checkpoint selection, and experiment logging;
5. benchmark end-to-end latency in addition to model-only latency;
6. add ROC/EER reporting and exportable result tables;
7. run the same full protocol for all three models.
