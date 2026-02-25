"""Temporal contrastive losses used for SimCLR training."""

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


class TemporalAllPairsTTLoss(nn.Module):
    """
    Multi-positive temporal InfoNCE over full windows.

    Input embeddings are expected as [B, T, D], where:
      - B = number of windows in a batch
      - T = number of frames per window
      - D = embedding dimension

    For each anchor frame, every other frame from the same window is a positive
    (for example: a1-a2, a1-a3, ...). Frames from other windows are negatives.
    """

    def __init__(self, temperature: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"Expected z to be (B, T, D), got {tuple(z.shape)}")

        bsz, t_steps, dim = z.shape
        if t_steps < 2:
            raise ValueError(f"T must be >= 2 to form temporal positives, got T={t_steps}")

        z = F.normalize(z.reshape(bsz * t_steps, dim), dim=1, eps=self.eps)
        logits = (z @ z.T) / self.temperature

        n = bsz * t_steps
        idx = torch.arange(n, device=z.device)
        self_mask = idx[:, None] == idx[None, :]

        # Sample ids identify which window each frame belongs to.
        sample_ids = torch.arange(bsz, device=z.device).repeat_interleave(t_steps)
        same_sample = sample_ids[:, None] == sample_ids[None, :]

        pos_mask = same_sample & (~self_mask)
        denom_mask = ~self_mask

        logits_for_denom = logits.masked_fill(~denom_mask, float("-inf"))
        log_denom = torch.logsumexp(logits_for_denom, dim=1, keepdim=True)
        log_prob = logits - log_denom

        # Average over all (ordered) positive temporal pairs in the batch.
        loss = -log_prob[pos_mask].mean()
        return loss
