"""Physical bases for SphereNet: radial, circular and full spherical.

The three representations follow the paper indexing (Liu et al., ICLR 2022):

- ``Psi(d)``           width ``N``: trainable sine radial expansion with the
  smooth polynomial cutoff envelope.
- ``Psi(d, theta)``    width ``L*N``: spherical Bessel ``j_l`` paired with the
  real harmonic ``Y_l0``, ``l = 0..L-1``, ``n = 0..N-1``.
- ``Psi(d, theta, phi)`` width ``L^2*N``: every real harmonic ``Y_lm``,
  ``m = -l..+l``, paired with the Bessel radial function of the *same degree*
  ``l``.

The discrete spherical-Bessel roots and harmonic normalizers are constructed
once (SciPy at construction time); every forward operation is pure PyTorch and
uses stable recurrences so first and second coordinate derivatives are finite.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Envelope(nn.Module):
    """DIG-style smooth polynomial cutoff ``u(x)`` with the ``1/x`` factor."""

    def __init__(self, exponent: int) -> None:
        super().__init__()
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 1:
            raise ValueError("envelope exponent must be a positive integer")
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, scaled: Tensor) -> Tensor:
        """Return ``1/x + a*x^(p-1) + b*x^p + c*x^(p+1)`` for ``x`` in (0, 1]."""

        x_pow_p0 = scaled.pow(self.p - 1)
        x_pow_p1 = x_pow_p0 * scaled
        x_pow_p2 = x_pow_p1 * scaled
        return 1.0 / scaled + self.a * x_pow_p0 + self.b * x_pow_p1 + self.c * x_pow_p2


class RadialBasis(nn.Module):
    """``Psi(d)``: trainable sine expansion with the polynomial cutoff."""

    def __init__(
        self,
        num_radial: int,
        *,
        cutoff: float,
        envelope_exponent: int = 5,
    ) -> None:
        super().__init__()
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.envelope = Envelope(envelope_exponent)
        self.freq = nn.Parameter(torch.empty(num_radial))
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.num_radial

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.freq.copy_(
                torch.arange(1, self.num_radial + 1, dtype=self.freq.dtype)
                * math.pi
            )

    def forward(self, distances: Tensor) -> Tensor:
        """Expand ``[Q]`` distances to ``[Q, N]``."""

        _validate_vector(distances, "distances")
        scaled = distances.unsqueeze(-1) / self.cutoff
        frequency = self.freq.to(dtype=distances.dtype)
        return self.envelope(scaled) * torch.sin(frequency * scaled)


class CircularBasis(nn.Module):
    """``Psi(d, theta)``: Bessel ``j_l`` times the real harmonic ``Y_l0``."""

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        *,
        cutoff: float,
    ) -> None:
        super().__init__()
        _positive_int(num_spherical, "num_spherical")
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        roots, normalizers = _spherical_bessel_roots(num_spherical, num_radial)
        self.register_buffer("roots", roots)
        self.register_buffer("normalizers", normalizers)
        self._normalizers = _harmonic_normalizers(num_spherical)

    @property
    def output_dim(self) -> int:
        return self.num_spherical * self.num_radial

    def forward(self, distances: Tensor, cos_theta: Tensor) -> Tensor:
        """Return ``[Q, L*N]`` with columns ordered ``(l, n)`` l-major."""

        _validate_vector(distances, "distances")
        _validate_vector(cos_theta, "cos_theta")
        if cos_theta.shape != distances.shape:
            raise ValueError("distances and cos_theta must share the [Q] shape")
        if cos_theta.device != distances.device:
            raise ValueError("distances and cos_theta must share a device")
        scaled = distances.unsqueeze(-1) / self.cutoff
        legendre = _associated_legendre_table(cos_theta, self.num_spherical - 1)
        blocks: list[Tensor] = []
        for order in range(self.num_spherical):
            args = scaled * self.roots[order]
            radial = self.normalizers[order] * _spherical_bessel(order, args)
            angular = self._normalizers[(order, 0)] * legendre[order][0]
            blocks.append(radial * angular.unsqueeze(-1))
        return torch.cat(blocks, dim=-1)


class TorsionBasis(nn.Module):
    """``Psi(d, theta, phi)``: every real harmonic paired with same-degree Bessel.

    Column order is paper ``(l, m, n)``: for ``l = 0..L-1``, ``m = -l..+l``,
    ``n = 0..N-1``.  The legacy DIG "flatten then ``view(L, L)``" layout is
    deliberately not reproduced because it does not retain the degree pairing.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        *,
        cutoff: float,
    ) -> None:
        super().__init__()
        _positive_int(num_spherical, "num_spherical")
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        roots, normalizers = _spherical_bessel_roots(num_spherical, num_radial)
        self.register_buffer("roots", roots)
        self.register_buffer("normalizers", normalizers)
        self._normalizers = _harmonic_normalizers(num_spherical)

    @property
    def output_dim(self) -> int:
        return self.num_spherical * self.num_spherical * self.num_radial

    def forward(
        self, distances: Tensor, cos_theta: Tensor, phi: Tensor
    ) -> Tensor:
        """Return ``[Q, L^2*N]`` with columns ordered ``(l, m, n)``."""

        _validate_vector(distances, "distances")
        _validate_vector(cos_theta, "cos_theta")
        _validate_vector(phi, "phi")
        if not (cos_theta.shape == phi.shape == distances.shape):
            raise ValueError(
                "distances, cos_theta and phi must share the [Q] shape"
            )
        if not (phi.device == distances.device == cos_theta.device):
            raise ValueError("distances, cos_theta and phi must share a device")
        scaled = distances.unsqueeze(-1) / self.cutoff
        legendre = _associated_legendre_table(cos_theta, self.num_spherical - 1)
        blocks: list[Tensor] = []
        for order in range(self.num_spherical):
            args = scaled * self.roots[order]
            radial = self.normalizers[order] * _spherical_bessel(order, args)
            for m in range(-order, order + 1):
                angular = _harmonic_value(
                    order, m, legendre, phi, self._normalizers
                )
                blocks.append(radial * angular.unsqueeze(-1))
        return torch.cat(blocks, dim=-1)


def _harmonic_value(
    order: int,
    m: int,
    legendre_table: list[list[Tensor]],
    phi: Tensor,
    normalizers: dict[tuple[int, int], float],
) -> Tensor:
    absolute = abs(m)
    angular = normalizers[(order, m)] * legendre_table[order][absolute]
    if m == 0:
        return angular
    if m > 0:
        return angular * torch.cos(absolute * phi)
    return angular * torch.sin(absolute * phi)


def _harmonic_normalizers(
    num_spherical: int,
) -> dict[tuple[int, int], float]:
    normalizers: dict[tuple[int, int], float] = {}
    for order in range(num_spherical):
        for m in range(-order, order + 1):
            absolute = abs(m)
            factor = (
                (2 * order + 1)
                * math.factorial(order - absolute)
                / (4 * math.pi * math.factorial(order + absolute))
            )
            base = math.sqrt(factor)
            normalizers[(order, m)] = base * math.sqrt(2.0) if m != 0 else base
    return normalizers


def _associated_legendre_table(
    cos_theta: Tensor, max_degree: int
) -> list[list[Tensor]]:
    """Legendre/associated-Legendre values ``P_l^m(cos_theta)`` with phase.

    Returns ``table[l][m]`` for ``0 <= m <= l <= max_degree`` including the
    Condon-Shortley factor ``(-1)^m``.
    """

    sin_theta = torch.sqrt(torch.clamp(1 - cos_theta.square(), min=0.0))
    table: list[list[Tensor]] = [[None] * (l + 1) for l in range(max_degree + 1)]
    table[0][0] = torch.ones_like(cos_theta)
    for m in range(1, max_degree + 1):
        table[m][m] = -(2 * m - 1) * sin_theta * table[m - 1][m - 1]
    for m in range(max_degree):
        if m + 1 <= max_degree:
            table[m + 1][m] = (2 * m + 1) * cos_theta * table[m][m]
    for m in range(max_degree + 1):
        for l in range(m + 2, max_degree + 1):
            table[l][m] = (
                (2 * l - 1) * cos_theta * table[l - 1][m]
                - (l + m - 1) * table[l - 2][m]
            ) / (l - m)
    return table


def _spherical_bessel_roots(
    num_spherical: int,
    num_radial: int,
) -> tuple[Tensor, Tensor]:
    """Normalized positive roots ``j_l(beta_ln) = 0`` via SciPy at construction."""

    try:
        import numpy as np
        from scipy.optimize import brentq
        from scipy.special import spherical_jn
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "SphereNet requires SciPy when constructing its spherical basis; "
            "install molgnn-lab[spherenet]."
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
        next_order = spherical_jn(order + 1, order_roots)
        normalizers[order] = np.sqrt(2.0 / np.square(next_order))
    return (
        torch.from_numpy(roots.astype(np.float32)),
        torch.from_numpy(normalizers.astype(np.float32)),
    )


def _positive_spherical_bessel_roots(
    order: int,
    count: int,
    brentq: object,
    spherical_jn: object,
    np: object,
):
    """Numerically find the first positive roots without an import-time table."""

    upper = (count + order / 2 + 2) * math.pi
    roots: list[float] = []
    while len(roots) < count:
        samples = max(1_024, math.ceil(upper * 64))
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
    """Evaluate ``j_l`` with stable behavior around the origin.

    A short power series covers small arguments; Miller's downward recurrence
    covers everything else.  Both paths are pure torch operations.
    """

    if values.numel() == 0:
        return values
    small = values.abs() < 0.5
    series = _spherical_bessel_series(order, values)

    calculation_dtype = (
        torch.float64 if values.dtype in {torch.float16, torch.float32} else values.dtype
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
    for term_index in range(1, 12):
        term = term * (-squared / (2 * term_index * (2 * order + 2 * term_index + 1)))
        series = series + term
    return series


def _odd_double_factorial(value: int) -> float:
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return float(result)


def _validate_vector(values: Tensor, name: str) -> None:
    if (
        not isinstance(values, Tensor)
        or values.ndim != 1
        or not torch.is_floating_point(values)
    ):
        raise ValueError(f"{name} must have shape [Q] and be a floating tensor")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


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


__all__ = [
    "CircularBasis",
    "Envelope",
    "RadialBasis",
    "TorsionBasis",
]
