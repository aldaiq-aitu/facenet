from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model_registry import build_model, get_model_spec
from training.data import FaceImageFolder
from training.losses import ArcMarginProduct


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face-recognition backbone with ArcFace.")
    parser.add_argument("--model", choices=["facenet", "mobilefacenet", "efficientnet_lite0"], required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
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

    return total_loss / total_samples, total_correct / total_samples


def main():
    args = parse_args()
    spec = get_model_spec(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FaceImageFolder(args.train_dir, input_size=spec.input_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model).to(device)
    head = ArcMarginProduct(spec.embedding_size, num_classes=len(dataset.classes)).to(device)
    optimizer = AdamW([*model.parameters(), *head.parameters()], lr=args.lr)

    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, head, loader, optimizer, device)
        print(f"epoch={epoch:03d} loss={loss:.4f} train_acc={acc:.4f}")

    checkpoint = {
        "model_name": args.model,
        "model_state_dict": model.state_dict(),
        "arcface_head_state_dict": head.state_dict(),
        "classes": dataset.classes,
        "input_size": spec.input_size,
        "embedding_size": spec.embedding_size,
    }
    output_path = args.output / f"{args.model}_arcface.pt"
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
