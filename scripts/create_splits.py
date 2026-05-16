from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create identity-disjoint train/val/test splits.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--min-images-per-identity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_identities(root: Path, min_images: int):
    identities = []
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = [path for path in identity_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        if len(images) >= min_images:
            identities.append((identity_dir, images))
    return identities


def copy_identity(identity_dir: Path, images: list[Path], destination_root: Path):
    target_dir = destination_root / identity_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, target_dir / image.name)


def main():
    args = parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    identities = collect_identities(args.input_dir, args.min_images_per_identity)
    if len(identities) < 3:
        raise ValueError("Need at least three eligible identities to create train/val/test splits.")

    rng = random.Random(args.seed)
    rng.shuffle(identities)

    total = len(identities)
    if total < 5:
        raise ValueError("Need at least five eligible identities for train/val/test verification workflow.")

    val_count = max(2, round(total * args.val_ratio))
    test_count = max(2, total - round(total * args.train_ratio) - val_count)
    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError("Split ratios leave no training identities.")

    train_end = train_count
    val_end = train_count + val_count
    splits = {
        "train": identities[:train_end],
        "val": identities[train_end:val_end],
        "test": identities[val_end:],
    }

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for split_name, split_identities in splits.items():
        split_root = args.output_dir / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        for identity_dir, images in split_identities:
            copy_identity(identity_dir, images, split_root)
            manifest_rows.append([split_name, identity_dir.name, len(images)])

    manifest_path = args.output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "identity", "num_images"])
        writer.writerows(manifest_rows)

    print(
        "Created identity-disjoint splits: "
        + ", ".join(f"{name}={len(items)} identities" for name, items in splits.items())
    )
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
