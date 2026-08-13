"""Paper-oriented Fourier--Bessel basis functions for DimeNet-2020.

The public DimeNet profile fixes a five Angstrom cutoff, but the basis
modules keep the cutoff as an explicit construction argument so their
mathematics is independently testable.  The discrete spherical-Bessel roots
are constructed once, while every forward operation remains in PyTorch and
therefore differentiable with respect to the supplied distances.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .constants import DIMENET_CUTOFF, DIMENET_ENVELOPE_P


_SMALL_DISTANCE = 1e-4


class CutoffEnvelope(nn.Module):
    """The polynomial cutoff ``u(r / c)`` from DimeNet Equation (8).

    This is deliberately only the polynomial envelope.  In particular, it
    does not carry a hidden inverse-distance factor: Equation (7) already
    contains the radial ``1 / r`` term, whereas the spherical basis does not.
    """

    def __init__(self, envelope_p: int = DIMENET_ENVELOPE_P) -> None:
        super().__init__()
        _positive_int(envelope_p, "envelope_p")
        self.envelope_p = envelope_p
        self._coefficient_p = (envelope_p + 1) * (envelope_p + 2) / 2
        self._coefficient_p1 = envelope_p * (envelope_p + 2)
        self._coefficient_p2 = envelope_p * (envelope_p + 1) / 2

    def forward(self, scaled_distances: Tensor) -> Tensor:
        """Return the smooth compact-support multiplier for ``r / c``."""

        if not isinstance(scaled_distances, Tensor) or not torch.is_floating_point(
            scaled_distances
        ):
            raise ValueError("scaled_distances must be a floating tensor")
        if not torch.isfinite(scaled_distances).all():
            raise ValueError("scaled_distances must contain only finite values")

        distance = scaled_distances
        polynomial = (
            1
            - self._coefficient_p * distance.pow(self.envelope_p)
            + self._coefficient_p1 * distance.pow(self.envelope_p + 1)
            - self._coefficient_p2 * distance.pow(self.envelope_p + 2)
        )
        return torch.where(distance < 1, polynomial, torch.zeros_like(polynomial))


class RadialBesselBasis(nn.Module):
    """DimeNet's enveloped radial Fourier--Bessel basis (Equation (7))."""

    def __init__(
        self,
        num_radial: int,
        *,
        cutoff: float = DIMENET_CUTOFF,
        envelope_p: int = DIMENET_ENVELOPE_P,
    ) -> None:
        super().__init__()
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.envelope = CutoffEnvelope(envelope_p)
        self.frequencies = nn.Parameter(torch.empty(num_radial))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the fine-tunable frequencies to the ``n*pi`` roots."""

        with torch.no_grad():
            self.frequencies.copy_(
                torch.arange(
                    1,
                    self.num_radial + 1,
                    device=self.frequencies.device,
                    dtype=self.frequencies.dtype,
                )
                * math.pi
            )

    def forward(self, distances: Tensor) -> Tensor:
        """Expand ``[E]`` distances to a ``[E, num_radial]`` basis tensor."""

        _validate_distances(distances, "distances")
        scaled = distances / self.cutoff
        frequencies = self.frequencies.to(dtype=distances.dtype)
        arguments = scaled.unsqueeze(-1) * frequencies
        small = distances.abs() < _SMALL_DISTANCE
        safe_distances = torch.where(small, torch.ones_like(distances), distances)
        regular = torch.sin(arguments) / safe_distances.unsqueeze(-1)

        # sin(k r / c) / r has a finite analytic limit at r = 0.  Retaining
        # the low-order series keeps the basis finite and differentiable for
        # direct callers even though valid radius edges never have zero length.
        limit = (frequencies / self.cutoff) * (
            1 - arguments.square() / 6 + arguments.pow(4) / 120
        )
        radial = torch.where(small.unsqueeze(-1), limit, regular)
        return math.sqrt(2 / self.cutoff) * self.envelope(scaled).unsqueeze(-1) * radial


class SphericalBesselBasis(nn.Module):
    """The joint spherical Fourier--Bessel basis from DimeNet Equation (6)."""

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        *,
        cutoff: float = DIMENET_CUTOFF,
        envelope_p: int = DIMENET_ENVELOPE_P,
    ) -> None:
        super().__init__()
        _positive_int(num_spherical, "num_spherical")
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.envelope = CutoffEnvelope(envelope_p)

        roots, normalizers = _spherical_bessel_roots(
            num_spherical, num_radial, self.cutoff
        )
        self.register_buffer("roots", roots)
        self.register_buffer("normalizers", normalizers)

    @property
    def output_dim(self) -> int:
        """Return the flattened ``l``-by-``n`` basis width."""

        return self.num_spherical * self.num_radial

    def forward(self, distances: Tensor, angles: Tensor) -> Tensor:
        """Return ``[Q, num_spherical * num_radial]`` joint features.

        ``distances`` must already correspond to incoming ``k -> j`` edges and
        ``angles`` to the matching ``k -> j -> i`` triplets.  Keeping that
        indexing outside the basis module makes the two DimeNet distance
        branches explicit in the model's interaction call.
        """

        _validate_distances(distances, "distances")
        if (
            not isinstance(angles, Tensor)
            or angles.ndim != 1
            or not torch.is_floating_point(angles)
        ):
            raise ValueError("angles must have shape [Q] and be a floating tensor")
        if angles.shape != distances.shape:
            raise ValueError("distances and angles must have the same shape")
        if angles.device != distances.device:
            raise ValueError("distances and angles must share a device")
        if not torch.isfinite(angles).all():
            raise ValueError("angles must contain only finite values")

        return self.forward_from_cosine(distances, torch.cos(angles))

    def forward_from_cosine(self, distances: Tensor, cosines: Tensor) -> Tensor:
        """Evaluate the same ``m=0`` basis directly from ``cos(angle)``.

        Real spherical harmonics with ``m=0`` are Legendre polynomials of the
        angle cosine.  This form is exactly equivalent to :meth:`forward` for
        valid angles, while avoiding an unnecessary inverse-angle operation in
        coordinate-gradient paths, including legal collinear triplets.
        """

        _validate_distances(distances, "distances")
        if (
            not isinstance(cosines, Tensor)
            or cosines.ndim != 1
            or not torch.is_floating_point(cosines)
        ):
            raise ValueError("cosines must have shape [Q] and be a floating tensor")
        if cosines.shape != distances.shape:
            raise ValueError("distances and cosines must have the same shape")
        if cosines.device != distances.device:
            raise ValueError("distances and cosines must share a device")
        if not torch.isfinite(cosines).all():
            raise ValueError("cosines must contain only finite values")

        scaled = distances / self.cutoff
        envelope = self.envelope(scaled).unsqueeze(-1)
        blocks: list[Tensor] = []
        for order in range(self.num_spherical):
            roots = self.roots[order].to(dtype=distances.dtype)
            arguments = scaled.unsqueeze(-1) * roots
            radial = _spherical_bessel(order, arguments)
            normalizer = self.normalizers[order].to(dtype=distances.dtype)
            angular = _real_spherical_harmonic_m0_from_cosine(order, cosines).unsqueeze(
                -1
            )
            blocks.append(envelope * normalizer * radial * angular)
        return torch.cat(blocks, dim=-1)


def _spherical_bessel_roots(
    num_spherical: int,
    num_radial: int,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Build normalized positive roots of spherical Bessel functions.

    SciPy is intentionally imported only here.  It is an optional DimeNet
    construction dependency, never an import-time dependency of the package
    or a forward-time dependency of the model.
    """

    try:
        import numpy as np
        from scipy.optimize import brentq
        from scipy.special import spherical_jn
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "DimeNet requires SciPy when constructing SphericalBesselBasis; "
            "install molgnn-lab[dimenet]."
        ) from exc

    roots = np.empty((num_spherical, num_radial), dtype=np.float64)
    normalizers = np.empty_like(roots)
    for order in range(num_spherical):
        if order == 0:
            order_roots = np.arange(1, num_radial + 1, dtype=np.float64) * math.pi
        else:
            order_roots = _positive_spherical_bessel_roots(
                order, num_radial, brentq, spherical_jn, np
            )
        roots[order] = order_roots
        next_order_values = spherical_jn(order + 1, order_roots)
        normalizers[order] = np.sqrt(2 / (cutoff**3 * np.square(next_order_values)))
    return torch.from_numpy(roots.astype(np.float32)), torch.from_numpy(
        normalizers.astype(np.float32)
    )


def _positive_spherical_bessel_roots(
    order: int,
    count: int,
    brentq: object,
    spherical_jn: object,
    np: object,
):
    """Numerically find the first positive roots without an import-time table."""

    # The nth root approaches ``(n + l / 2) * pi``.  A dense sign scan is
    # robust for the small basis sizes exposed by DimeNet and grows only when
    # a caller requests an unusually large order/count pair.
    upper = (count + order / 2 + 2) * math.pi
    roots: list[float] = []
    while len(roots) < count:
        samples = max(1_024, int(math.ceil(upper * 64)))
        grid = np.linspace(1e-7, upper, samples, dtype=np.float64)
        values = spherical_jn(order, grid)
        sign_changes = np.flatnonzero(values[:-1] * values[1:] < 0)
        roots = []
        for index in sign_changes:
            left = float(grid[index])
            right = float(grid[index + 1])
            root = float(brentq(lambda value: spherical_jn(order, value), left, right))
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(root)
            if len(roots) == count:
                break
        if len(roots) < count:
            upper *= 2
    return np.asarray(roots, dtype=np.float64)


def _spherical_bessel(order: int, values: Tensor) -> Tensor:
    """Evaluate ``j_l`` with stable series behavior around the origin."""

    # Upward recurrence is numerically unstable for low arguments and higher
    # angular orders.  Use a short power series near zero and Miller's
    # *downward* recurrence elsewhere.  Both paths are pure torch operations,
    # retaining first and second coordinate derivatives in the model.
    small = values.abs() < 0.5
    series = _spherical_bessel_series(order, values)
    if values.numel() == 0:
        return values

    calculation_dtype = (
        torch.float64
        if values.dtype in {torch.float16, torch.float32}
        else values.dtype
    )
    calculation_values = values.to(dtype=calculation_dtype)
    safe_values = torch.where(
        small, torch.ones_like(calculation_values), calculation_values
    )
    max_argument = int(torch.ceil(safe_values.detach().abs().max()).item())
    start_order = max(order + 32, max_argument + 32)

    following = torch.zeros_like(safe_values)
    current = torch.ones_like(safe_values)
    selected: Tensor | None = current if order == start_order else None
    for degree in range(start_order, 0, -1):
        previous = (2 * degree + 1) * current / safe_values - following
        if degree - 1 == order:
            selected = previous
        following, current = current, previous
    assert selected is not None
    j0 = torch.sin(safe_values) / safe_values
    regular = selected * j0 / current
    return torch.where(small, series, regular.to(dtype=values.dtype))


def _spherical_bessel_series(order: int, values: Tensor) -> Tensor:
    """Stable power series for ``j_l`` over small arguments."""

    term = values.pow(order) / _odd_double_factorial(2 * order + 1)
    series = term
    squared = values.square()
    # Twelve terms are far below float32 error for |x| < 0.5, while the
    # explicit loop retains a differentiable expression rather than a lookup.
    for term_index in range(1, 12):
        term = term * (-squared / (2 * term_index * (2 * order + 2 * term_index + 1)))
        series = series + term
    return series


def _real_spherical_harmonic_m0_from_cosine(order: int, cosine: Tensor) -> Tensor:
    """Evaluate the real ``Y_l^0`` spherical harmonic using Legendre recursion."""

    if order == 0:
        legendre = torch.ones_like(cosine)
    elif order == 1:
        legendre = cosine
    else:
        previous = torch.ones_like(cosine)
        current = cosine
        for degree in range(2, order + 1):
            following = (
                (2 * degree - 1) * cosine * current - (degree - 1) * previous
            ) / degree
            previous, current = current, following
        legendre = current
    return math.sqrt((2 * order + 1) / (4 * math.pi)) * legendre


def _odd_double_factorial(value: int) -> float:
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return float(result)


def _validate_distances(distances: Tensor, name: str) -> None:
    if (
        not isinstance(distances, Tensor)
        or distances.ndim != 1
        or not torch.is_floating_point(distances)
    ):
        raise ValueError(f"{name} must have shape [Q] and be a floating tensor")
    if not torch.isfinite(distances).all():
        raise ValueError(f"{name} must contain only finite values")
    if distances.numel() and torch.any(distances < 0):
        raise ValueError(f"{name} must be non-negative")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


__all__ = ["CutoffEnvelope", "RadialBesselBasis", "SphericalBesselBasis"]
