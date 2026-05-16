# Dataset Strategy

## What the project requires

The training pipeline expects a folder-per-identity dataset:

```text
raw_dataset/
  identity_001/
    image_001.jpg
    image_002.jpg
  identity_002/
    image_001.jpg
```

For a defensible experiment, use:

- many identities;
- multiple images per identity;
- natural variation in pose, lighting, and expression;
- permission to use the data for academic work.

## Recommended strategy

### Stronger research route

Use a large face-recognition training dataset that you are legally allowed to
download, then run the full repository pipeline and evaluate on a held-out split
or an external verification benchmark.

### Practical route if access is limited

Use a smaller legally available identity-labeled face dataset for training and
make the limitation explicit in the diploma. The repository keeps train,
validation, and test identities disjoint so the methodology remains honest even
when the dataset is smaller.

## Recommended fast thesis dataset

For the time-constrained diploma workflow, use a controlled CelebA subset:

```text
1000 identities x 20 images = 20,000 images
```

Build it from the official CelebA aligned images and identity annotations:

```bash
python -m scripts.prepare_celeba_subset \
  --images-dir /path/to/img_align_celeba \
  --identity-file /path/to/identity_CelebA.txt \
  --output-dir data/celeba_light \
  --num-identities 1000 \
  --images-per-identity 20 \
  --min-images-per-identity 20
```

Then use `data/celeba_light` as the input to the normal alignment/split/pair
workflow. Since CelebA already provides aligned images, you may use it directly
as the split input when schedule matters, or run `scripts.align_dataset` anyway
if you want the exact same alignment stage recorded in your experiment log.

## Why the repository does not auto-download a dataset

Large face datasets often have separate license terms, registration steps, or
distribution restrictions. The code is therefore dataset-agnostic by design:
you provide a legally obtained dataset, and the repository performs alignment,
splitting, pair generation, training, evaluation, and plotting.
