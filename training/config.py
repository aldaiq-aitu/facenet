from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    model: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    arcface_scale: float
    arcface_margin: float
    scheduler: str
    min_lr: float
    early_stopping_patience: int
    monitor: str
    train_dir: Path
    val_pairs_csv: Path
    output_dir: Path


def load_train_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(
        model=raw["model"],
        seed=int(raw["seed"]),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        num_workers=int(raw["num_workers"]),
        arcface_scale=float(raw["arcface_scale"]),
        arcface_margin=float(raw["arcface_margin"]),
        scheduler=str(raw["scheduler"]),
        min_lr=float(raw["min_lr"]),
        early_stopping_patience=int(raw["early_stopping_patience"]),
        monitor=str(raw["monitor"]),
        train_dir=Path(raw["train_dir"]),
        val_pairs_csv=Path(raw["val_pairs_csv"]),
        output_dir=Path(raw["output_dir"]),
    )
