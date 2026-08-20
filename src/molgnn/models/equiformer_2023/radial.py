"""Distance expansions used by the Gaussian Equiformer core."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import Tensor, nn


class GaussianRadialBasis(nn.Module):
    """Learnable Gaussian basis matching the author's QM9 source path."""

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        if num_basis < 1:
            raise ValueError("num_basis must be positive")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        self.num_basis = num_basis
        self.cutoff = float(cutoff)
        self.mean = nn.Parameter(torch.empty(1, num_basis))
        self.std = nn.Parameter(torch.empty(1, num_basis))
        self.weight = nn.Parameter(torch.ones(1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1))
        nn.init.uniform_(self.mean, 0.0, 1.0)
        nn.init.uniform_(self.std, 1.0 / num_basis, 1.0)

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        normalized = distances.unsqueeze(-1) / self.cutoff
        normalized = self.weight * normalized + self.bias
        normalized = normalized.expand(-1, self.num_basis)
        std = self.std.abs() + 1e-5
        return torch.exp(-0.5 * ((normalized - self.mean) / std).square()) / (
            math.sqrt(2.0 * math.pi) * std
        )


class RadialProfile(nn.Module):
    """The source's Linear--LayerNorm--SiLU radial MLP with final offset."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) < 2 or any(channel < 1 for channel in channels):
            raise ValueError("radial profile needs at least two positive channel counts")
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(pairwise(channels)):
            final = index == len(channels) - 2
            layers.append(nn.Linear(in_features, out_features, bias=not final))
            if not final:
                layers.extend((nn.LayerNorm(out_features), nn.SiLU()))
        self.net = nn.Sequential(*layers)
        self.offset = nn.Parameter(torch.empty(channels[-1]))
        bound = 1.0 / math.sqrt(channels[-2])
        nn.init.uniform_(self.offset, -bound, bound)

    def apply_output_scale(self, scale: Tensor) -> None:
        """Match source fan-in initialization for per-edge tensor-product weights."""

        if scale.shape != self.offset.shape:
            raise ValueError("radial output scale must match the profile output width")
        final_linear = self.net[-1]
        if not isinstance(final_linear, nn.Linear):  # pragma: no cover - construction invariant
            raise TypeError("radial profile must end with nn.Linear")
        with torch.no_grad():
            final_linear.weight.mul_(scale[:, None])
            self.offset.mul_(scale)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features) + self.offset


__all__ = ["GaussianRadialBasis", "RadialProfile"]
