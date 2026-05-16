from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create balanced verification pairs from one split.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--positive-pairs", type=int, default=1000)
    parser.add_argument("--negative-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_images(root: Path):
    identities = {}
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = [path for path in identity_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        if len(images) >= 2:
            identities[identity_dir.name] = images
    return identities


def main():
    args = parse_args()
    identities = collect_images(args.split_dir)
    if len(identities) < 2:
        raise ValueError("Need at least two identities with two images each.")

    rng = random.Random(args.seed)
    positive_candidates = [
        (first, second, 1)
        for images in identities.values()
        for first, second in itertools.combinations(images, 2)
    ]
    if not positive_candidates:
        raise ValueError("No positive pair candidates found.")

    identity_items = list(identities.items())
    negative_candidates = []
    for (name_a, images_a), (name_b, images_b) in itertools.combinations(identity_items, 2):
        for first in images_a:
            for second in images_b:
                negative_candidates.append((first, second, 0))

    positives = rng.sample(positive_candidates, k=min(args.positive_pairs, len(positive_candidates)))
    negatives = rng.sample(negative_candidates, k=min(args.negative_pairs, len(negative_candidates)))
    rows = positives + negatives
    rng.shuffle(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path1", "path2", "is_same"])
        writer.writerows([[str(a), str(b), label] for a, b, label in rows])

    print(
        f"Saved {len(positives)} positive and {len(negatives)} negative pairs "
        f"to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
