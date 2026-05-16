# Energy-Efficient Face Recognition System

A diploma-oriented face-recognition project that combines:

- a real-time webcam demo;
- a reproducible research pipeline for training and comparing FaceNet,
  MobileFaceNet, and EfficientNet-Lite0 under one protocol.

## Repository layout

```text
.
├── configs/                  # experiment configs for all three models
├── evaluation/               # verification metrics and reports
├── models/                   # backbone implementations
├── scripts/                  # alignment, splits, pairs, batch experiments
├── training/                 # ArcFace training pipeline
├── benchmark_checkpoints.py  # benchmark trained checkpoints
├── camera.py                 # real-time webcam application
├── embeddings.py             # runtime embeddings and recognition
├── preprocessing.py          # shared alignment and normalization
├── COLAB_GUIDE.md            # GPU execution guide
├── DATASETS.md               # dataset strategy
├── RESEARCH_PIPELINE.md      # implementation details
└── THESIS_WORKFLOW.md         # diploma structure and required artifacts
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Real-time demo

```bash
python main.py
```

Controls:

| Key | Action |
|---|---|
| `E` | enroll a face |
| `D` | delete a person |
| `L` | list enrolled people |
| `Q` | quit |

## Full research workflow

```bash
python -m scripts.align_dataset --input-dir raw_dataset --output-dir data/aligned
python -m scripts.create_splits --input-dir data/aligned --output-dir data/splits
python -m scripts.create_pairs --split-dir data/splits/val --output-csv data/pairs/val_pairs.csv
python -m scripts.create_pairs --split-dir data/splits/test --output-csv data/pairs/test_pairs.csv
python -m scripts.run_all_experiments
python -m evaluation.compare_models \
  --pairs-csv data/pairs/test_pairs.csv \
  --checkpoint experiments/facenet/best.pt \
  --checkpoint experiments/mobilefacenet/best.pt \
  --checkpoint experiments/efficientnet_lite0/best.pt \
  --output-csv results/model_comparison.csv
python benchmark_checkpoints.py \
  --checkpoint experiments/facenet/best.pt \
  --checkpoint experiments/mobilefacenet/best.pt \
  --checkpoint experiments/efficientnet_lite0/best.pt \
  --output-csv results/benchmark_checkpoints.csv
python plot_results.py \
  --metrics-csv results/model_comparison.csv \
  --benchmark-csv results/benchmark_checkpoints.csv
```

## Compared models

| Model | Input | Embedding | Role |
|---|---:|---:|---|
| `facenet` | 160x160 | 512-d | accuracy-oriented baseline |
| `mobilefacenet` | 112x112 | 128-d | lightweight candidate |
| `efficientnet_lite0` | 112x112 | 512-d | efficient CNN candidate |

## Research outputs

The pipeline produces:

- aligned-face manifest;
- identity-disjoint train/validation/test splits;
- balanced verification-pair CSVs;
- training histories;
- best and last checkpoints;
- validation-calibrated thresholds;
- held-out test metrics;
- ROC curves;
- checkpoint benchmarks;
- thesis-ready comparison plots.

## Supporting documents

- [COLAB_GUIDE.md](COLAB_GUIDE.md)
- [notebooks/train_all_models_colab.ipynb](notebooks/train_all_models_colab.ipynb)
- [DATASETS.md](DATASETS.md)
- [MODEL_USAGE.md](MODEL_USAGE.md)
- [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md)
- [THESIS_WORKFLOW.md](THESIS_WORKFLOW.md)

## Notes

- Local biometric databases (`*.pkl`) are intentionally ignored by Git.
- The repository is dataset-agnostic because many large face datasets have
  separate license or registration requirements.
- The local machine used for development may run the demo on CPU, while final
  training is expected to run on a GPU environment such as Google Colab.
