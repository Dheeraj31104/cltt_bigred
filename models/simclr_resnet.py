"""SimCLR backbone and projection head models."""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class SimCLRResNet18(nn.Module):
    """ResNet-18 encoder with a SimCLR-style projection head."""

    def __init__(self, proj_dim: int = 128):
        super().__init__()

        base = models.resnet18(weights=None)
        num_ftrs = base.fc.in_features  # 512 for ResNet-18
        base.fc = nn.Identity()

        self.encoder = base
        self.feature_dim = num_ftrs
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z
