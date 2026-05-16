from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED_PATHS = [
    Path("data/splits"),
    Path("data/pairs/val_pairs.csv"),
    Path("data/pairs/test_pairs.csv"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Package shared experiment data for multiple devices.")
    parser.add_argument("--output-dir", type=Path, default=Path("shared_experiment_bundle"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required_paths(output_dir: Path):
    for path in REQUIRED_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Required path is missing: {path}")
        destination = output_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)


def build_manifest(output_dir: Path):
    entries = []
    for file_path in sorted(path for path in output_dir.rglob("*") if path.is_file()):
        entries.append(
            {
                "path": str(file_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": sha256_file(file_path),
                "bytes": file_path.stat().st_size,
            }
        )
    return {"files": entries}


def main():
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copy_required_paths(args.output_dir)
    manifest = build_manifest(args.output_dir)
    manifest_path = args.output_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Shared experiment bundle created at {args.output_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
