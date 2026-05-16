# Project Overview

## 1. Project Summary

This project is a local, real-time face recognition application that captures webcam frames, detects faces with MTCNN, extracts embeddings with a selectable recognition backbone, and compares them against a local pickle database. It also includes research scripts for benchmarking model speed/size and plotting benchmark metrics.

Tech stack and versions:
- Python: documented as 3.11 in `README.md`; syntax checked locally with Python 3.14.2.
- PyTorch: `torch==2.2.2` documented in `README.md`.
- TorchVision: `torchvision==0.17.2` documented in `README.md`.
- OpenCV: `opencv-contrib-python==4.8.1.78` documented in `README.md`.
- facenet-pytorch: used for MTCNN and InceptionResnetV1; no pinned version in the project.
- NumPy: documented as `>=1.24.0,<2.0.0`.
- Pillow: used for image conversion/cropping; no pinned version in the project.
- timm: used by EfficientNet-Lite0 and `download_weights.py`; no pinned version in the project.
- psutil: used by `benchmark.py`; no pinned version in the project.
- matplotlib, seaborn, pandas: used by plotting scripts; no pinned versions in the project.

Directory structure:

```text
Diploma/
├── benchmark.py
├── benchmark_results.txt
├── camera.py
├── config.py
├── database.py
├── download_weights.py
├── embeddings.py
├── facenet_recognition.py
├── faces_database.pkl
├── faces_database_backup_512d.pkl
├── image_2026-01-25_23-07-15.png
├── main.py
├── MODEL_USAGE.md
├── models/
│   ├── __init__.py
│   ├── efficientnet_lite.py
│   ├── facenet.py
│   └── mobilefacenet.py
├── plot_benchmark.py
├── plot_metrics.py
├── PROJECT_OVERVIEW.md
├── README.md
└── requirements.txt
```

## 2. Architecture

The application is a single-process desktop/webcam program. `main.py` imports and starts the camera loop in `camera.py`. `camera.py` owns webcam capture, keyboard interactions, display rendering, background detection submission, OpenCV tracking, and calls into the database and embedding layers. `embeddings.py` initializes the global device, MTCNN detector, selected recognition model, embedding extraction helper, and cosine-similarity recognizer. `database.py` persists face embeddings in a local pickle file and stores a `__backbone__` metadata key to avoid mixing embeddings from incompatible models. `config.py` is the central configuration module.

Data flow:

```text
Webcam frame
  -> BGR/RGB conversion in camera.py
  -> AsyncDetector background MTCNN detection
  -> PIL face crop
  -> get_embedding_from_crop() in embeddings.py
  -> recognize() compares against database.py data
  -> label/color drawn on OpenCV window
```

Key files and responsibilities:
- `main.py`: CLI entry point.
- `camera.py`: OpenCV loop, key controls, enrollment/deletion/listing interactions, async detection, tracker updates.
- `embeddings.py`: detector/model initialization, input preprocessing, embedding extraction, cosine recognition.
- `facenet_recognition.py`: standalone FaceNet/VGGFace2 webcam recognizer with enrollment, deletion, listing, and a FaceNet-only pickle database.
- `database.py`: load/save/add/delete/list helpers for pickle-backed embedding storage.
- `config.py`: backbone, camera cadence, recognition threshold, database path, FaceNet model name.
- `models/facenet.py`: wrapper around `facenet_pytorch.InceptionResnetV1`.
- `models/mobilefacenet.py`: MobileNetV2-based lightweight embedding model.
- `models/efficientnet_lite.py`: timm EfficientNet-Lite0 embedding model.
- `benchmark.py`: model-only latency, parameter, memory, CPU benchmark runner.
- `plot_benchmark.py`: generates PNG charts from benchmark values.
- `plot_metrics.py`: generates a theoretical metric chart.
- `download_weights.py`: pre-downloads MobileNetV2 and EfficientNet-Lite0 weights.
- `requirements.txt`: installable dependency manifest.

## 3. Known Issues (Found During Analysis)

- `README.md` documents only `E` and `Q` controls, but the app also implements `D` delete and `L` list.
- `README.md` documents stale defaults: `FRAME_SKIP=5`, `RECOG_INTERVAL=5`, and `MIN_FACE_SIZE=80`; actual values in `config.py` are `15`, `15`, and `120`.
- No dependency manifest exists (`requirements.txt`, `pyproject.toml`, or similar), so setup depends on manually copied commands from `README.md`.
- `camera.py` uses `cv2.legacy.TrackerMOSSE_create()` directly. This fails on OpenCV builds where the contrib legacy namespace is absent or where tracker factory names differ.
- `camera.py` checks `if not database` in `remove_face()`, but the database always contains `__backbone__`, so deletion still prompts even when no people are enrolled.
- `camera.py` slices tracker crops without clamping the lower and upper bounds consistently. Negative or out-of-frame tracker values can produce invalid/empty crops and repeated embedding errors.
- `AsyncDetector.submit()` reads and writes `_busy` without a lock, so rapid frame submissions can race with worker startup/completion.
- `AsyncDetector.submit()` stores `_frame_rgb` without copying it, unlike `_frame_bgr`; this is fragile if the source frame is reused or mutated.
- `embeddings.py` swallows embedding exceptions by printing only to stdout and returning `None`; it does not include traceback context.
- `recognize()` recomputes the normalized query embedding inside the inner loop for every stored vector.
- `download_weights.py` catches download errors and continues, then prints "All downloads finished" even if one or both downloads failed.
- `benchmark.py` hard-codes "Runs per model: 200" in saved output even when `benchmark()` is called with a different run count.
- `plot_benchmark.py` uses hard-coded benchmark numbers instead of parsing `benchmark_results.txt`, so generated charts can become stale.
- `plot_metrics.py` imports NumPy only to create an unused `x` variable.
- `lol.js` appears unrelated to the Python face-recognition project and references external globals (`API`, `UTILS`, `dataUUID`, `documentID`) that are undefined in this repository.
- There are no automated tests.
- No API integrations are used by the main app. External network calls only happen implicitly when pretrained model weights are downloaded by PyTorch/timm/facenet-pytorch.

Console/runtime checks:
- `python3 -m py_compile` succeeded for all Python files.
- Webcam runtime was not executed during audit because it requires camera access and an interactive OpenCV window.
- Benchmark runtime was not executed during audit because it can download large model weights and take minutes.

API calls:
- The main Python application has no HTTP API calls.
- `download_weights.py` triggers model weight downloads through TorchVision and timm.
- `lol.js` calls platform-specific `API.*` methods, but that script is not wired into the Python project.

## 4. Dependencies & Environment

Required environment variables:
- None required by the Python application.

External services/APIs:
- Local webcam device via OpenCV (`cv2.VideoCapture(0)`).
- PyTorch, TorchVision, facenet-pytorch, and timm model weight hosts when pretrained weights are not already cached.
- Local filesystem for `faces_database.pkl`.

Setup instructions:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Optional research commands:

```bash
python benchmark.py
python plot_benchmark.py
python plot_metrics.py
python download_weights.py
```

## 5. Change Log

- `camera.py` — fixed empty-delete prompt, thread-safe detector submission, copied RGB frames for async work, robust OpenCV tracker creation, and clamped tracker crops.
- `embeddings.py` — added traceback logging for embedding failures and normalized query embeddings once per recognition call.
- `requirements.txt` — added a dependency manifest covering the runtime, benchmark, plotting, and weight-download scripts.
- `benchmark.py` — made saved benchmark output report the actual configured run count instead of a hard-coded value.
- `download_weights.py` — made failed model downloads raise an error and changed the completion message to success-only.
- `download_weights.py` — removed an unused `torch` import.
- `plot_benchmark.py` — replaced stale hard-coded chart metrics with parsing from `benchmark_results.txt`.
- `plot_benchmark.py` — removed an unused loop variable from chart annotations.
- `plot_metrics.py` — removed an unused NumPy import and unused `x` variable.
- `README.md` — aligned setup instructions, keyboard controls, and configuration defaults with the current code.
- `main.py` — configured application logging so runtime errors logged by modules are visible.
- `lol.js` — removed unrelated untracked JavaScript automation code that was not part of the Python face-recognition project.
- `facenet_recognition.py` — added a complete standalone FaceNet/VGGFace2 recognition script with webcam detection, enrollment, deletion, listing, and local database persistence.
- `README.md` — documented the recommended FaceNet-only launch command.
