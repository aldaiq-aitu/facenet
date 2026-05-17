# Three-Device Training Plan

Use this plan when three different machines are available and time is limited.

## Roles

| Device | Assign | Reason |
|---|---|---|
| Device A: strongest GPU | `facenet` | heaviest model |
| Device B: medium GPU | `efficientnet_lite0` | medium workload |
| Device C: weakest GPU | `mobilefacenet` | lightest model |

## Non-negotiable rule

All three devices must use:

1. the same Git commit;
2. the same prepared dataset subset;
3. the same `data/splits/`;
4. the same `data/pairs/val_pairs.csv`;
5. the same `data/pairs/test_pairs.csv`.

Do **not** regenerate splits independently on each device. Prepare once, then
copy the same bundle everywhere.

## Step 1 — prepare data once on the coordinator machine

```bash
python -m scripts.prepare_celeba_subset \
  --images-dir /path/to/img_align_celeba \
  --identity-file /path/to/identity_CelebA.txt \
  --output-dir data/celeba_light \
  --num-identities 1000 \
  --images-per-identity 20 \
  --min-images-per-identity 20

python -m scripts.create_splits \
  --input-dir data/celeba_light \
  --output-dir data/splits

python -m scripts.create_pairs \
  --split-dir data/splits/val \
  --output-csv data/pairs/val_pairs.csv

python -m scripts.create_pairs \
  --split-dir data/splits/test \
  --output-csv data/pairs/test_pairs.csv

python -m scripts.package_experiment_bundle \
  --output-dir shared_experiment_bundle
```

Copy `shared_experiment_bundle/` to all three devices, preserving its internal
folder layout.

## Step 2 — set up each device

On every device:

```bash
git clone https://github.com/aldaiq-aitu/facenet.git
cd facenet
git checkout codex/research-foundation
pip install -r requirements.txt
```

Then copy the bundle contents into the repository root so each device has:

```text
data/splits/
data/pairs/val_pairs.csv
data/pairs/test_pairs.csv
```

## Step 3 — train exactly one model per device

### Device A — strongest GPU

```bash
python -m scripts.run_device_experiment --model facenet
```

### Device B — medium GPU

```bash
python -m scripts.run_device_experiment --model efficientnet_lite0
```

### Device C — weakest GPU

```bash
python -m scripts.run_device_experiment --model mobilefacenet
```

## Step 4 — return artifacts to the coordinator

From each device collect:

```text
experiments/<model>/best.pt
experiments/<model>/history.csv
results/<model>/metrics.json
results/<model>/scores.csv
results/<model>/roc_curve.png
results/<model>/benchmark.csv
```

## Step 5 — merge final results on the coordinator

Place all three checkpoint folders under:

```text
experiments/facenet/
experiments/mobilefacenet/
experiments/efficientnet_lite0/
```

If FaceNet is used as a pretrained baseline instead of being trained, first run:

```bash
python -m scripts.evaluate_pretrained_baseline \
  --val-pairs-csv data/pairs/val_pairs.csv \
  --test-pairs-csv data/pairs/test_pairs.csv \
  --output-dir baselines/facenet_pretrained
```

Then use:

```text
baselines/facenet_pretrained/baseline.pt
```

in place of `experiments/facenet/best.pt`.

Then run:

```bash
python -m evaluation.compare_models \
  --pairs-csv data/pairs/test_pairs.csv \
  --checkpoint baselines/facenet_pretrained/baseline.pt \
  --checkpoint experiments/mobilefacenet/best.pt \
  --checkpoint experiments/efficientnet_lite0/best.pt \
  --output-csv results/model_comparison.csv

python benchmark_checkpoints.py \
  --checkpoint baselines/facenet_pretrained/baseline.pt \
  --checkpoint experiments/mobilefacenet/best.pt \
  --checkpoint experiments/efficientnet_lite0/best.pt \
  --output-csv results/benchmark_checkpoints.csv

python plot_results.py \
  --metrics-csv results/model_comparison.csv \
  --benchmark-csv results/benchmark_checkpoints.csv
```

## Expected final outputs

```text
results/model_comparison.csv
results/benchmark_checkpoints.csv
results/plots/*.png
```

These are the files you will use in the diploma tables and figures.
