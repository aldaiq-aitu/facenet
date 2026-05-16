"""
FaceNet — Heavyweight, highly accurate face recognition model.

Paper:  FaceNet: A Unified Embedding for Face Recognition and Clustering
        (arXiv:1503.03832)

This wrapper uses the facenet-pytorch implementation (InceptionResnetV1) 
with pretrained weights (VGGFace2 by default).

Input:  160×160 RGB face crop
Output: 512-dimensional embedding vector
Params: ~27.9 M 
"""

import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceNet(nn.Module):
    """
    Wrapper around InceptionResnetV1 from facenet_pytorch.
    Provides a consistent interface with other models in this project.
    """

    def __init__(self, pretrained='vggface2'):
        super().__init__()
        # Load the pretrained FaceNet model
        self.model = InceptionResnetV1(pretrained=pretrained)

    def forward(self, x):
        return self.model(x)
