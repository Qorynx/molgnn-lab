"""Distance and angle bases used by the official HMGNN implementation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DistanceRBF(nn.Module):
    """Gaussian basis over ``exp(-distance)`` from the official source."""

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        _positive_int(num_basis, "num_basis")
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        cutoff_value = float(cutoff)
        centers = torch.linspace(math.exp(-cutoff_value), 1.0, num_basis)
        width = ((2.0 / num_basis) * (1.0 - math.exp(-cutoff_value))) ** -2
        self.register_buffer("centers", centers)
        self.register_buffer("beta", torch.tensor(width, dtype=torch.float32))

    def forward(self, distance: Tensor) -> Tensor:
        return torch.exp(-self.beta * (torch.exp(-distance.unsqueeze(-1)) - self.centers) ** 2)


class ShrinkDistanceRBF(DistanceRBF):
    """Distance basis with HMGNN's quintic cutoff envelope."""

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__(num_basis, cutoff)
        self.cutoff = float(cutoff)

    def forward(self, distance: Tensor) -> Tensor:
        radial = super().forward(distance)
        ratio = distance / self.cutoff
        envelope = 1.0 - 10.0 * ratio**3 + 15.0 * ratio**4 - 6.0 * ratio**5
        return radial * envelope.unsqueeze(-1)


class AngleRBF(nn.Module):
    """Direct Gaussian expansion of angles on ``[0, pi]``."""

    def __init__(self, num_basis: int) -> None:
        super().__init__()
        _positive_int(num_basis, "num_basis")
        centers = torch.linspace(0.0, math.pi, num_basis)
        width = ((2.0 / num_basis) * math.pi) ** -2
        self.register_buffer("centers", centers)
        self.register_buffer("beta", torch.tensor(width, dtype=torch.float32))

    def forward(self, angle: Tensor) -> Tensor:
        return torch.exp(-self.beta * (angle.unsqueeze(-1) - self.centers) ** 2)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = ["AngleRBF", "DistanceRBF", "ShrinkDistanceRBF"]
