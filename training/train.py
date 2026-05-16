from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from evaluation.inference import score_pairs
from evaluation.io import read_pairs_csv
from evaluation.metrics import find_best_threshold
from model_registry import build_model, get_model_spec
from training.config import TrainConfig, load_train_config
from training.data import FaceImageFolder
from training.losses import ArcMarginProduct
from training.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face-recognition backbone with ArcFace.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def train_one_epoch(model, head, loader, optimizer, device):
    model.train()
    head.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        logits = head(embeddings, labels)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def save_checkpoint(path: Path, cfg: TrainConfig, spec, model, head, classes, metrics):
    checkpoint = {
        "model_name": cfg.model,
        "model_state_dict": model.state_dict(),
        "arcface_head_state_dict": head.state_dict(),
        "classes": classes,
        "input_size": spec.input_size,
        "embedding_size": spec.embedding_size,
        "validation_metrics": metrics.to_dict(),
    }
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)

    spec = get_model_spec(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = FaceImageFolder(cfg.train_dir, input_size=spec.input_size, training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_pairs = read_pairs_csv(cfg.val_pairs_csv)

    model = build_model(cfg.model).to(device)
    head = ArcMarginProduct(
        spec.embedding_size,
        num_classes=len(train_dataset.classes),
        scale=cfg.arcface_scale,
        margin=cfg.arcface_margin,
    ).to(device)
    optimizer = AdamW(
        [*model.parameters(), *head.parameters()],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = cfg.output_dir / "history.csv"
    best_path = cfg.output_dir / "best.pt"
    last_path = cfg.output_dir / "last.pt"
    config_copy_path = cfg.output_dir / "config.json"
    config_copy_path.write_text(json.dumps({**cfg.__dict__}, default=str, indent=2), encoding="utf-8")

    best_value = float("-inf")
    epochs_without_improvement = 0

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_accuracy",
                "val_f1",
                "val_far",
                "val_frr",
                "val_auc",
                "val_eer",
                "threshold",
                "learning_rate",
            ]
        )

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(model, head, train_loader, optimizer, device)
            scores, labels = score_pairs(model, val_pairs, spec.input_size, device)
            val_metrics = find_best_threshold(scores, labels)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_accuracy,
                    val_metrics.accuracy,
                    val_metrics.f1,
                    val_metrics.far,
                    val_metrics.frr,
                    val_metrics.roc_auc,
                    val_metrics.eer,
                    val_metrics.threshold,
                    current_lr,
                ]
            )
            file.flush()

            print(
                f"epoch={epoch:03d} loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f} "
                f"eer={val_metrics.eer:.4f} threshold={val_metrics.threshold:.4f}"
            )

            monitored_value = getattr(val_metrics, cfg.monitor.removeprefix("val_"))
            if monitored_value > best_value:
                best_value = monitored_value
                epochs_without_improvement = 0
                save_checkpoint(best_path, cfg, spec, model, head, train_dataset.classes, val_metrics)
            else:
                epochs_without_improvement += 1

            save_checkpoint(last_path, cfg, spec, model, head, train_dataset.classes, val_metrics)
            scheduler.step()

            if epochs_without_improvement >= cfg.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"Best checkpoint saved to {best_path}")
    print(f"Last checkpoint saved to {last_path}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
