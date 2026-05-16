# Project Overview

## 1. Purpose

The project implements a real-time face-recognition system and a reproducible
research workflow for comparing three neural-network backbones:

- FaceNet;
- MobileFaceNet;
- EfficientNet-Lite0.

The target diploma question is not only whether the system works, but which
backbone offers the best trade-off between verification quality and computational
efficiency.

## 2. Runtime architecture

```text
webcam frame
  -> MTCNN detection
  -> landmark-based face alignment
  -> backbone embedding
  -> cosine similarity against local database
  -> identity label in the OpenCV window
```

Main runtime files:

- `main.py`
- `camera.py`
- `embeddings.py`
- `database.py`
- `preprocessing.py`

## 3. Research architecture

```text
raw identity folders
  -> scripts.align_dataset
  -> scripts.create_splits
  -> scripts.create_pairs
  -> training.train with ArcFace
  -> evaluation.compare_models
  -> benchmark_checkpoints
  -> plot_results
```

The research workflow deliberately keeps:

- aligned inputs shared across models;
- train/validation/test identities disjoint;
- the training objective shared;
- validation threshold calibration separate from test evaluation;
- benchmark methodology identical across trained checkpoints.

## 4. Core modules

| Area | Files |
|---|---|
| Models | `models/*`, `model_registry.py` |
| Preprocessing | `preprocessing.py`, `scripts/align_dataset.py` |
| Data preparation | `scripts/create_splits.py`, `scripts/create_pairs.py` |
| Training | `training/*`, `configs/*` |
| Evaluation | `evaluation/*` |
| Benchmarking | `benchmark.py`, `benchmark_checkpoints.py`, `plot_results.py` |
| Documentation | `README.md`, `RESEARCH_PIPELINE.md`, `THESIS_WORKFLOW.md`, `COLAB_GUIDE.md`, `DATASETS.md` |

## 5. Current engineering status

Implemented:

- real-time webcam recognition demo;
- shared model registry;
- shared normalization;
- five-point face alignment;
- MobileFaceNet-style lightweight backbone;
- ArcFace training head;
- identity-disjoint data splitting;
- verification-pair generation;
- validation-aware training with early stopping;
- held-out evaluation with FAR/FRR/AUC/EER;
- trained-checkpoint benchmarking;
- thesis-plot generation;
- smoke tests for core components.

Still requiring external resources:

- a legally usable face dataset;
- a GPU runtime for full training;
- final measured experimental results for the diploma text.

## 6. Reproducibility outputs

Every full experiment should preserve:

- aligned data manifest;
- split manifest;
- validation and test pair CSVs;
- experiment config copy;
- training history;
- best checkpoint;
- model comparison CSV;
- benchmark CSV;
- plots.
