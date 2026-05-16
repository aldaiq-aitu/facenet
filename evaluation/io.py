from __future__ import annotations

import csv
from pathlib import Path


def read_pairs_csv(path: str | Path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"path1", "path2", "is_same"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"pairs CSV must contain exactly columns: {sorted(required)}")
        for row in reader:
            rows.append((Path(row["path1"]), Path(row["path2"]), int(row["is_same"])))
    return rows
