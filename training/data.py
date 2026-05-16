from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from preprocessing import prepare_face_tensor


class FaceImageFolder(Dataset):
    """ImageFolder wrapper with shared train-time augmentation."""

    def __init__(self, root: str | Path, input_size: int, training: bool):
        self.dataset = ImageFolder(root=str(root))
        self.input_size = input_size
        self.training = training
        self.augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            ]
        )

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image_path, label = self.dataset.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.input_size, self.input_size))
        if self.training:
            image = self.augment(image)
        return prepare_face_tensor(image), label
