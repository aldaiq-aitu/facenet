from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from preprocessing import align_face, crop_face


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Align a folder-per-identity face dataset.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def iter_images(root: Path):
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield identity_dir.name, image_path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(keep_all=True, device=device, min_face_size=args.min_face_size)
    manifest_path = args.manifest or (args.output_dir / "alignment_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total = aligned_count = fallback_count = skipped_count = 0

    for identity, image_path in iter_images(args.input_dir):
        total += 1
        try:
            image = Image.open(image_path).convert("RGB")
            frame_rgb = np.asarray(image)
            boxes, probs, landmarks = detector.detect(image, landmarks=True)
        except Exception as exc:
            rows.append([identity, str(image_path), "", "error", str(exc)])
            skipped_count += 1
            continue

        if boxes is None or probs is None:
            rows.append([identity, str(image_path), "", "no_face", ""])
            skipped_count += 1
            continue

        best_idx = int(np.argmax(probs))
        if float(probs[best_idx]) < args.min_confidence:
            rows.append([identity, str(image_path), "", "low_confidence", f"{float(probs[best_idx]):.4f}"])
            skipped_count += 1
            continue

        aligned = align_face(frame_rgb, landmarks[best_idx], args.image_size)
        if aligned is None:
            fallback = crop_face(frame_rgb, boxes[best_idx], args.image_size)
            if fallback is None:
                rows.append([identity, str(image_path), "", "invalid_crop", ""])
                skipped_count += 1
                continue
            face = fallback.image
            status = "fallback_crop"
            fallback_count += 1
        else:
            face = aligned.image
            status = "aligned"
            aligned_count += 1

        relative_name = image_path.stem + ".jpg"
        output_path = args.output_dir / identity / relative_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        face.save(output_path, quality=95)
        rows.append([identity, str(image_path), str(output_path), status, f"{float(probs[best_idx]):.4f}"])

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["identity", "source_path", "output_path", "status", "confidence_or_error"])
        writer.writerows(rows)

    print(
        f"Processed={total} aligned={aligned_count} fallback={fallback_count} "
        f"skipped={skipped_count} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
