"""Sparse PyTorch layers for the 2016 Weave architecture.

The original architecture carries an atom state and an ordered atom-pair
state in parallel.  ``weave_pair_index[0]`` is deliberately the atom that
receives the pair-to-atom aggregate.  This convention keeps the sparse pair
representation equivalent to the source pair matrix while avoiding a dense
``[num_atoms, num_atoms, pair_dim]`` allocation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Batch normalization that remains defined for one atom or pair.

    Molecular batches can legitimately contain a single atom or a single
    pair.  PyTorch's training-mode batch normalization rejects that case, so
    use the accumulated running statistics without mutating them.  Empty pair
    tensors are also valid for callers that supply a sparse pair graph.
    """

    def forward(self, values: Tensor) -> Tensor:
        """Normalize ``values`` without failing on a singleton leading axis."""

        if values.numel() == 0:
            return values
        if self.training and values.shape[0] <= 1:
            return F.batch_norm(
                values,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(values)


class WeaveModule(nn.Module):
    """One simultaneous atom/pair update from the Weave architecture.

    The four source operations are implemented explicitly:

    * atom-to-atom (``A -> A``),
    * pair-to-atom (``P -> A``), summed by the first pair endpoint,
    * pair-to-pair (``P -> P``), and
    * symmetric atom-to-pair (``A -> P``), ``f(a, b) + f(b, a)``.

    All per-item learned transforms use a linear map, optional batch
    normalization, and ReLU.  Pair state is updated rather than discarded so
    consecutive modules retain the source architecture's coupled states.
    """

    def __init__(
        self,
        atom_in_dim: int,
        pair_in_dim: int,
        hidden_dim: int,
        atom_out_dim: int | None = None,
        pair_out_dim: int | None = None,
        *,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_in_dim, "atom_in_dim"),
            (pair_in_dim, "pair_in_dim"),
            (hidden_dim, "hidden_dim"),
        ):
            _positive_int(value, name)
        if atom_out_dim is None:
            atom_out_dim = hidden_dim
        if pair_out_dim is None:
            pair_out_dim = hidden_dim
        _positive_int(atom_out_dim, "atom_out_dim")
        _positive_int(pair_out_dim, "pair_out_dim")
        if not isinstance(batch_norm, bool):
            raise ValueError("batch_norm must be a boolean")

        self.atom_in_dim = atom_in_dim
        self.pair_in_dim = pair_in_dim
        self.hidden_dim = hidden_dim
        self.atom_out_dim = atom_out_dim
        self.pair_out_dim = pair_out_dim

        self.atom_to_atom = nn.Linear(atom_in_dim, hidden_dim)
        self.pair_to_atom = nn.Linear(pair_in_dim, hidden_dim)
        self.update_atom = nn.Linear(2 * hidden_dim, atom_out_dim)

        self.pair_to_pair = nn.Linear(pair_in_dim, hidden_dim)
        self.atom_to_pair = nn.Linear(2 * atom_in_dim, hidden_dim)
        self.update_pair = nn.Linear(2 * hidden_dim, pair_out_dim)

        self.atom_to_atom_norm = _normalization(hidden_dim, batch_norm)
        self.pair_to_atom_norm = _normalization(hidden_dim, batch_norm)
        self.update_atom_norm = _normalization(atom_out_dim, batch_norm)
        self.pair_to_pair_norm = _normalization(hidden_dim, batch_norm)
        self.atom_to_pair_norm = _normalization(hidden_dim, batch_norm)
        self.update_pair_norm = _normalization(pair_out_dim, batch_norm)

    def forward(
        self, atom_states: Tensor, pair_states: Tensor, pair_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update atom and ordered-pair states.

        Args:
            atom_states: ``[N, atom_in_dim]`` atom state tensor.
            pair_states: ``[Q, pair_in_dim]`` ordered-pair state tensor.
            pair_index: ``[2, Q]`` pair endpoints.  The first row identifies
                the atom to which pair-to-atom messages are summed.
        """

        source, target = pair_index
        atom_to_atom = _relu(self.atom_to_atom_norm(self.atom_to_atom(atom_states)))

        pair_to_atom_messages = _relu(
            self.pair_to_atom_norm(self.pair_to_atom(pair_states))
        )
        pair_to_atom = atom_states.new_zeros(
            (atom_states.shape[0], self.hidden_dim)
        )
        if pair_to_atom_messages.shape[0]:
            pair_to_atom.index_add_(0, source, pair_to_atom_messages)
        next_atoms = _relu(
            self.update_atom_norm(
                self.update_atom(torch.cat((atom_to_atom, pair_to_atom), dim=-1))
            )
        )

        pair_to_pair = _relu(self.pair_to_pair_norm(self.pair_to_pair(pair_states)))
        atom_to_pair = self._symmetric_atom_to_pair(atom_states, source, target)
        next_pairs = _relu(
            self.update_pair_norm(
                self.update_pair(torch.cat((pair_to_pair, atom_to_pair), dim=-1))
            )
        )
        return next_atoms, next_pairs

    def _symmetric_atom_to_pair(
        self, atom_states: Tensor, source: Tensor, target: Tensor
    ) -> Tensor:
        """Return the source operation ``f(a, b) + f(b, a)`` for each pair.

        Evaluate both endpoint orders in one concatenated batch.  Besides
        retaining exact pair-order symmetry, this makes batch normalization
        use one shared set of batch statistics during training.
        """

        ordered = torch.cat((atom_states[source], atom_states[target]), dim=-1)
        reversed_order = torch.cat(
            (atom_states[target], atom_states[source]), dim=-1
        )
        pair_count = ordered.shape[0]
        both_orders = torch.cat((ordered, reversed_order), dim=0)
        transformed = _relu(
            self.atom_to_pair_norm(self.atom_to_pair(both_orders))
        )
        return transformed[:pair_count] + transformed[pair_count:]


class GaussianHistogramReadout(nn.Module):
    """Paper-style fixed 11-bin Gaussian histogram followed by graph sums.

    The histogram memberships are normalized across bins for every atom
    feature before they are summed by graph.  No post-histogram compression is
    applied: its ``output_dim`` is therefore ``input_dim * 11``.
    """

    _MEANS = (-1.645, -1.080, -0.739, -0.468, -0.228, 0.0, 0.228, 0.468, 0.739, 1.080, 1.645)
    _STDS = (0.283, 0.170, 0.134, 0.118, 0.114, 0.114, 0.114, 0.118, 0.134, 0.170, 0.283)

    def __init__(self, input_dim: int, gaussian_bins: int = 11) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        if gaussian_bins != len(self._MEANS):
            raise ValueError("gaussian_bins must be 11 for the fixed Weave histogram")

        self.input_dim = input_dim
        self.gaussian_bins = gaussian_bins
        self.output_dim = input_dim * gaussian_bins
        self.register_buffer("means", torch.tensor(self._MEANS, dtype=torch.float32))
        self.register_buffer("stds", torch.tensor(self._STDS, dtype=torch.float32))

    def gaussian_histogram(self, atom_states: Tensor) -> Tensor:
        """Expand ``[N, F]`` atom states to normalized ``[N, F * 11]`` bins."""

        normalized_distance = (
            atom_states.unsqueeze(-1) - self.means
        ) / self.stds
        memberships = torch.exp(-0.5 * normalized_distance.square())
        memberships = memberships + 1e-7
        memberships = memberships / memberships.sum(dim=-1, keepdim=True)
        return memberships.reshape(atom_states.shape[0], self.output_dim)

    def forward(
        self, atom_states: Tensor, graph_batch: Tensor, num_graphs: int
    ) -> Tensor:
        """Return one permutation-invariant histogram fingerprint per graph."""

        histogram = self.gaussian_histogram(atom_states)
        graph_features = atom_states.new_zeros((num_graphs, self.output_dim))
        graph_features.index_add_(0, graph_batch, histogram)
        return graph_features


def _normalization(num_features: int, enabled: bool) -> nn.Module:
    return SafeBatchNorm1d(num_features) if enabled else nn.Identity()


def _relu(values: Tensor) -> Tensor:
    return F.relu(values)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


WeaveLayer = WeaveModule


__all__ = [
    "GaussianHistogramReadout",
    "SafeBatchNorm1d",
    "WeaveLayer",
    "WeaveModule",
]
