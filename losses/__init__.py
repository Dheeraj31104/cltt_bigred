"""Loss functions for SimCLR training."""

from .paired_cosine_tt import PairedCosineTTLoss, TemporalAllPairsTTLoss

__all__ = ["PairedCosineTTLoss", "TemporalAllPairsTTLoss"]
