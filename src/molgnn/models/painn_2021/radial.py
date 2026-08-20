"""PaiNN radial bases and cosine cutoff."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .constants import PAINN_CUTOFF, PAINN_NUM_RBF


class BesselRBF(nn.Module):
    """The 20-term sine/Bessel basis used in the PaiNN paper."""

    def __init__(self, cutoff: float = PAINN_CUTOFF, num_rbf: int = PAINN_NUM_RBF) -> None:
        super().__init__()
        _positive_int(num_rbf, "num_rbf")
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.register_buffer(
            "frequencies",
            torch.arange(1, self.num_rbf + 1, dtype=torch.float32)
            * math.pi
            / self.cutoff,
        )

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        scaled = distances.unsqueeze(-1) * self.frequencies
        # sin(a*d)/d = a*sinc(a*d/pi), with the analytic d=0 limit.
        denominator = distances.unsqueeze(-1)
        values = torch.sin(scaled) / denominator.clamp_min(torch.finfo(distances.dtype).eps)
        zero = distances == 0
        if bool(zero.any()):
            values = torch.where(
                zero.unsqueeze(-1),
                self.frequencies.to(dtype=distances.dtype).expand_as(values),
                values,
            )
        return values


class GaussianRBF(nn.Module):
    """SchNetPack-compatible fixed Gaussian radial basis."""

    def __init__(self, cutoff: float = PAINN_CUTOFF, num_rbf: int = PAINN_NUM_RBF) -> None:
        super().__init__()
        _positive_int(num_rbf, "num_rbf")
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if num_rbf < 2:
            raise ValueError("num_rbf must be at least 2 for GaussianRBF")
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        offsets = torch.linspace(0.0, self.cutoff, self.num_rbf, dtype=torch.float32)
        widths = torch.full_like(offsets, float(offsets[1] - offsets[0]))
        self.register_buffer("offsets", offsets)
        self.register_buffer("widths", widths)

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        widths = self.widths.to(dtype=distances.dtype)
        offsets = self.offsets.to(dtype=distances.dtype)
        return torch.exp(-0.5 * ((distances.unsqueeze(-1) - offsets) / widths).square())


class CosineCutoff(nn.Module):
    """Behler-style cosine cutoff used by PaiNN."""

    def __init__(self, cutoff: float = PAINN_CUTOFF) -> None:
        super().__init__()
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)

    def forward(self, distances: Tensor) -> Tensor:
        value = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0)
        return value * (distances < self.cutoff).to(dtype=distances.dtype)


def build_radial_basis(
    name: str,
    *,
    cutoff: float = PAINN_CUTOFF,
    num_rbf: int = PAINN_NUM_RBF,
) -> nn.Module:
    """Construct a paper or SchNetPack-compatible radial basis."""

    if name == "bessel":
        return BesselRBF(cutoff=cutoff, num_rbf=num_rbf)
    if name == "gaussian":
        return GaussianRBF(cutoff=cutoff, num_rbf=num_rbf)
    raise ValueError("radial_basis must be either 'bessel' or 'gaussian'")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["BesselRBF", "CosineCutoff", "GaussianRBF", "build_radial_basis"]
