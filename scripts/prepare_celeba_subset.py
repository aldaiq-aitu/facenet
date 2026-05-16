from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a lightweight identity-balanced CelebA subset.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-identities", type=int, default=1000)
    parser.add_argument("--images-per-identity", type=int, default=20)
    parser.add_argument("--min-images-per-identity", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Use hardlink when source and output are on the same filesystem to save space.",
    )
    return parser.parse_args()


def read_identity_file(path: Path):
    identities = defaultdict(list)
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid identity annotation line: {raw_line!r}")
            image_name, identity_id = parts
            identities[identity_id].append(image_name)
    return identities


def materialize_image(source: Path, destination: Path, copy_mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def main():
    args = parse_args()
    if args.images_per_identity < 2:
        raise ValueError("images_per_identity must be at least 2.")
    if args.min_images_per_identity < args.images_per_identity:
        raise ValueError("min_images_per_identity must be >= images_per_identity.")

    image_map = read_identity_file(args.identity_file)
    eligible = {
        identity_id: image_names
        for identity_id, image_names in image_map.items()
        if len(image_names) >= args.min_images_per_identity
    }
    if len(eligible) < args.num_identities:
        raise ValueError(
            f"Requested {args.num_identities} identities, but only {len(eligible)} "
            f"have at least {args.min_images_per_identity} images."
        )

    rng = random.Random(args.seed)
    selected_identity_ids = rng.sample(sorted(eligible), k=args.num_identities)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    missing_files = []
    for new_index, identity_id in enumerate(selected_identity_ids, start=1):
        chosen_images = rng.sample(eligible[identity_id], k=args.images_per_identity)
        subset_identity = f"person_{new_index:04d}"
        for image_name in chosen_images:
            source = args.images_dir / image_name
            if not source.exists() or source.suffix.lower() not in IMAGE_SUFFIXES:
                missing_files.append(str(source))
                continue
            destination = args.output_dir / subset_identity / source.name
            materialize_image(source, destination, args.copy_mode)
            manifest_rows.append([subset_identity, identity_id, source.name, str(destination)])

    if missing_files:
        raise FileNotFoundError(
            "Some annotated CelebA images were missing from images_dir. "
            f"First missing file: {missing_files[0]}"
        )

    manifest_path = args.output_dir / "celeba_subset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["subset_identity", "celeba_identity_id", "image_name", "output_path"])
        writer.writerows(manifest_rows)

    print(
        f"Created CelebA-Light subset with {args.num_identities} identities x "
        f"{args.images_per_identity} images = {len(manifest_rows)} images."
    )
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
