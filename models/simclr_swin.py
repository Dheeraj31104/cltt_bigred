"""SimCLR with Swin Transformer (Tiny) backbone and projection head."""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import swin_t, swin_s, swin_b


_SWIN_VARIANTS = {
    "swin_t": (swin_t, 768),
    "swin_s": (swin_s, 768),
    "swin_b": (swin_b, 1024),
}


class SimCLRSwin(nn.Module):
    """Swin Transformer encoder with a SimCLR-style projection head.

    Supports swin_t (Tiny, feature_dim=768), swin_s (Small, feature_dim=768),
    and swin_b (Base, feature_dim=1024).
    """

    def __init__(self, variant: str = "swin_t", proj_dim: int = 128):
        super().__init__()
        if variant not in _SWIN_VARIANTS:
            raise ValueError(f"Unknown Swin variant '{variant}'. Choose from {list(_SWIN_VARIANTS)}")

        factory, num_ftrs = _SWIN_VARIANTS[variant]
        base = factory(weights=None)
        # torchvision Swin: the classification head is base.head (nn.Linear)
        base.head = nn.Identity()

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
