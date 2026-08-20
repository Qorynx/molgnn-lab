"""ViSNet's closed-form spherical features and vector normalization."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .constants import VISNET_EPS


def spherical_harmonics(directions: Tensor, lmax: int) -> Tensor:
    """Return the real l=1 or l=1,2 basis used by the supplied ViSNet source."""

    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape [E, 3]")
    if lmax not in {1, 2}:
        raise ValueError("lmax must be either 1 or 2")
    x, y, z = directions.unbind(dim=-1)
    first_order = (x, y, z)
    if lmax == 1:
        return torch.stack(first_order, dim=-1)

    second_order = (
        math.sqrt(3.0) * x * z,
        math.sqrt(3.0) * x * y,
        y.square() - 0.5 * (x.square() + z.square()),
        math.sqrt(3.0) * y * z,
        math.sqrt(3.0) / 2.0 * (z.square() - x.square()),
    )
    return torch.stack((*first_order, *second_order), dim=-1)


def vector_rejection(vector: Tensor, direction: Tensor) -> Tensor:
    """Remove the component of an ``[E, R, F]`` state along ``direction``."""

    if vector.ndim != 3 or direction.shape != vector.shape[:2]:
        raise ValueError("vector and direction must have shapes [E, R, F] and [E, R]")
    projection = (vector * direction.unsqueeze(-1)).sum(dim=1, keepdim=True)
    return vector - projection * direction.unsqueeze(-1)


class VecLayerNorm(nn.Module):
    """Source-compatible normalization for l=1 and l=1,2 vector states."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        trainable: bool = False,
        norm_type: str = "none",
        eps: float = VISNET_EPS,
    ) -> None:
        super().__init__()
        _positive_int(hidden_channels, "hidden_channels")
        if norm_type not in {"none", "rms", "max_min"}:
            raise ValueError("vecnorm_type must be one of 'none', 'rms', or 'max_min'")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.eps = float(eps)
        weight = torch.ones(hidden_channels, dtype=torch.float32)
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0)

    def forward(self, vector: Tensor) -> Tensor:
        if vector.ndim != 3 or vector.shape[-1] != self.hidden_channels:
            raise ValueError("vector must have shape [N, R, hidden_channels]")
        if vector.shape[1] == 3:
            normalized = self._normalize_block(vector)
        elif vector.shape[1] == 8:
            first, second = vector.split((3, 5), dim=1)
            normalized = torch.cat((self._normalize_block(first), self._normalize_block(second)), dim=1)
        else:
            raise ValueError("VecLayerNorm only supports 3 or 8 representation channels")
        return normalized * self.weight.to(dtype=vector.dtype).view(1, 1, -1)

    def _normalize_block(self, vector: Tensor) -> Tensor:
        if self.norm_type == "none":
            return vector
        distance = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
        if self.norm_type == "rms":
            scale = torch.sqrt(distance.square().mean(dim=-1, keepdim=True)).clamp_min(self.eps)
            return vector / scale

        direction = vector / distance.clamp_min(self.eps)
        maximum = distance.max(dim=-1, keepdim=True).values
        minimum = distance.min(dim=-1, keepdim=True).values
        span = (maximum - minimum).clamp_min(self.eps)
        return functional.relu((distance - minimum) / span) * direction


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["VecLayerNorm", "spherical_harmonics", "vector_rejection"]
