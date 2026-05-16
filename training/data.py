from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from preprocessing import prepare_face_tensor


class FaceImageFolder(Dataset):
    """ImageFolder wrapper returning normalized tensors and class labels."""

    def __init__(self, root: str | Path, input_size: int):
        self.dataset = ImageFolder(root=str(root))
        self.input_size = input_size

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image_path, label = self.dataset.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.input_size, self.input_size))
        return prepare_face_tensor(image), label
