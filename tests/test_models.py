import torch

from models.mobilefacenet import MobileFaceNet


def test_mobilefacenet_output_shape():
    model = MobileFaceNet()
    output = model(torch.randn(2, 3, 112, 112))
    assert output.shape == (2, 128)
