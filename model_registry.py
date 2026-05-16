from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_size: int
    embedding_size: int
    builder: Callable[[], nn.Module]


def _build_facenet() -> nn.Module:
    from models.facenet import FaceNet

    return FaceNet(pretrained="vggface2")


def _build_mobilefacenet() -> nn.Module:
    from models.mobilefacenet import MobileFaceNet

    return MobileFaceNet(embedding_size=128, input_size=112)


def _build_efficientnet_lite0() -> nn.Module:
    from models.efficientnet_lite import EfficientNetLite0Face

    return EfficientNetLite0Face(embedding_size=512, pretrained=True)


MODEL_SPECS = {
    "facenet": ModelSpec("facenet", 160, 512, _build_facenet),
    "mobilefacenet": ModelSpec("mobilefacenet", 112, 128, _build_mobilefacenet),
    "efficientnet_lite0": ModelSpec("efficientnet_lite0", 112, 512, _build_efficientnet_lite0),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc


def build_model(name: str) -> nn.Module:
    return get_model_spec(name).builder()
