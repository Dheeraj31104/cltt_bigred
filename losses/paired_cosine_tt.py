"""Temporal-pair cosine loss used for SimCLR training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairedCosineTTLoss(nn.Module):
    """Cosine similarity loss for consecutive positive pairs."""

    def __init__(self, temperature: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"Expected z to be (N, D), got {tuple(z.shape)}")
        n, _ = z.shape
        if n % 2 != 0:
            raise ValueError(f"N must be even (pairs), got N={n}")

        z = F.normalize(z, dim=1, eps=self.eps)
        sim = (z @ z.T) / self.temperature

        idx = torch.arange(n, device=z.device)
        pos_idx = idx ^ 1  # flips last bit: even->odd, odd->even

        pos_logits = sim[idx, pos_idx]

        neg_mask = torch.ones((n, n), dtype=torch.bool, device=z.device)
        neg_mask.fill_diagonal_(False)
        neg_mask[idx, pos_idx] = False

        neg_logits = sim[neg_mask].view(n, n - 2)

        loss_per = -(pos_logits - torch.logsumexp(neg_logits, dim=1))
        return loss_per.mean()
