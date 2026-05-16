# Google Colab Guide

This project is prepared so that the final training runs can be executed on a
GPU machine even when the local development machine has no CUDA device.

## 1. Open a GPU runtime

In Colab:

1. `Runtime` -> `Change runtime type`
2. choose `T4 GPU` or better

## 2. Clone the repository

```bash
!git clone https://github.com/aldaiq-aitu/facenet.git
%cd facenet
!pip install -r requirements.txt
```

## 3. Provide the raw dataset

Upload or mount a dataset arranged as:

```text
raw_dataset/
  identity_001/
    image_001.jpg
  identity_002/
    image_001.jpg
```

Recommended research-grade training choices:

- a large face-recognition dataset you are legally allowed to use;
- at least several images per identity;
- enough identities to keep validation and test identities disjoint from train identities.

## 4. Prepare aligned data

```bash
!python -m scripts.align_dataset \
  --input-dir /content/raw_dataset \
  --output-dir data/aligned \
  --image-size 112
```

## 5. Create identity-disjoint splits

```bash
!python -m scripts.create_splits \
  --input-dir data/aligned \
  --output-dir data/splits
```

## 6. Create validation and test pairs

```bash
!python -m scripts.create_pairs \
  --split-dir data/splits/val \
  --output-csv data/pairs/val_pairs.csv

!python -m scripts.create_pairs \
  --split-dir data/splits/test \
  --output-csv data/pairs/test_pairs.csv
```

## 7. Train all three models

```bash
!python -m scripts.run_all_experiments
```

Each experiment writes:

- `best.pt`
- `last.pt`
- `history.csv`
- `config.json`

under its own folder in `experiments/`.

## 8. Evaluate the final checkpoints on the held-out test set

```bash
!python -m evaluation.compare_models \
  --pairs-csv data/pairs/test_pairs.csv \
  --checkpoint experiments/facenet/best.pt \
  --checkpoint experiments/mobilefacenet/best.pt \
  --checkpoint experiments/efficientnet_lite0/best.pt \
  --output-csv results/model_comparison.csv
```

## 9. Benchmark trained checkpoints

```bash
!python benchmark_checkpoints.py \
  --checkpoint experiments/facenet/best.pt \
  --checkpoint experiments/mobilefacenet/best.pt \
  --checkpoint experiments/efficientnet_lite0/best.pt \
  --output-csv results/benchmark_checkpoints.csv
```

## 10. Generate thesis-ready plots

```bash
!python plot_results.py \
  --metrics-csv results/model_comparison.csv \
  --benchmark-csv results/benchmark_checkpoints.csv
```

## 11. Download the outputs

At minimum, save:

- `experiments/**/best.pt`
- `experiments/**/history.csv`
- `results/model_comparison.csv`
- `results/benchmark_checkpoints.csv`
- `results/plots/*.png`

These are the primary artifacts used in the diploma report.
