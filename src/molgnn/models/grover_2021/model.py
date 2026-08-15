"""Property-focused GROVER with parallel atom and directed-bond views."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import ADAPTATION_NAME


class _DynamicMPN(nn.Module):
    """One head-specific dynamic message-passing network for Q, K, or V."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        dropout: float,
        *,
        dynamic_depth: str,
        dense: bool,
        edge_dim: int | None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.depth = depth
        self.dropout = dropout
        self.dynamic_depth = dynamic_depth
        self.dense = dense
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.hidden_projection = nn.Linear(output_dim, output_dim)
        # The official implementation exposes the activation as a model
        # option and uses PReLU in the released pre-training configuration.
        # Keep a learnable activation here instead of hard-coding ReLU so the
        # local encoder follows that behavior while retaining its small API.
        self.activation = nn.PReLU(output_dim)
        self.edge_projection = (
            nn.Linear(edge_dim, output_dim) if edge_dim is not None else None
        )

    def forward(
        self,
        initial: Tensor,
        owner: Tensor,
        neighbor: Tensor,
        edge_ids: Tensor | None,
        edge_features: Tensor | None,
    ) -> Tensor:
        if initial.shape[0] == 0:
            return initial.new_empty((0, self.output_dim))
        state = self.input_projection(initial)
        initial_state = state
        num_steps = self._sample_depth()
        for _ in range(1, num_steps):
            messages = state.new_zeros(state.shape)
            if neighbor.numel():
                messages.index_add_(0, owner, state[neighbor])
                if (
                    edge_ids is not None
                    and edge_features is not None
                    and self.edge_projection is not None
                ):
                    messages.index_add_(
                        0, owner, self.edge_projection(edge_features[edge_ids])
                    )
            update = self.activation(initial_state + self.hidden_projection(messages))
            state = state + update if self.dense else update
            state = F.dropout(state, p=self.dropout, training=self.training)
        return state

    def _sample_depth(self) -> int:
        if not self.training or self.dynamic_depth == "none":
            return self.depth
        if self.dynamic_depth != "truncnorm":
            raise ValueError("dynamic_depth must be 'truncnorm' or 'none'")
        lower = max(1, self.depth - 3)
        upper = self.depth + 3
        draw = torch.normal(
            mean=float(self.depth),
            std=1.0,
            size=(),
            device=self.input_projection.weight.device,
        )
        return int(draw.round().clamp(lower, upper).item())


class _HeadwiseQKV(nn.Module):
    """Independent head-wise MPNs followed by standard scaled attention."""

    def __init__(
        self,
        hidden_dim: int,
        attention_hidden_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
        *,
        dynamic_depth: str,
        dense: bool,
        edge_dim: int | None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.query_heads = nn.ModuleList(
            _DynamicMPN(
                hidden_dim,
                attention_hidden_dim,
                depth,
                dropout,
                dynamic_depth=dynamic_depth,
                dense=dense,
                edge_dim=edge_dim,
            )
            for _ in range(num_heads)
        )
        self.key_heads = nn.ModuleList(
            _DynamicMPN(
                hidden_dim,
                attention_hidden_dim,
                depth,
                dropout,
                dynamic_depth=dynamic_depth,
                dense=dense,
                edge_dim=edge_dim,
            )
            for _ in range(num_heads)
        )
        self.value_heads = nn.ModuleList(
            _DynamicMPN(
                hidden_dim,
                attention_hidden_dim,
                depth,
                dropout,
                dynamic_depth=dynamic_depth,
                dense=dense,
                edge_dim=edge_dim,
            )
            for _ in range(num_heads)
        )
        self.output_projection = nn.Linear(
            num_heads * attention_hidden_dim, hidden_dim
        )

    def forward(
        self,
        values: Tensor,
        owner: Tensor,
        neighbor: Tensor,
        edge_ids: Tensor | None,
        edge_features: Tensor | None,
    ) -> Tensor:
        if values.shape[0] == 0:
            return values.new_empty(values.shape)
        queries = torch.stack(
            [
                head(values, owner, neighbor, edge_ids, edge_features)
                for head in self.query_heads
            ],
            dim=0,
        )
        keys = torch.stack(
            [
                head(values, owner, neighbor, edge_ids, edge_features)
                for head in self.key_heads
            ],
            dim=0,
        )
        values_by_head = torch.stack(
            [
                head(values, owner, neighbor, edge_ids, edge_features)
                for head in self.value_heads
            ],
            dim=0,
        )
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(
            self.attention_hidden_dim
        )
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, values_by_head).permute(1, 0, 2)
        return self.output_projection(attended.reshape(values.shape[0], -1))


class _MTBlock(nn.Module):
    """One parallel node-view and edge-view GTransformer block."""

    def __init__(
        self,
        hidden_dim: int,
        attention_hidden_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
        *,
        dynamic_depth: str,
        dense: bool,
        res_connection: bool,
    ) -> None:
        super().__init__()
        self.res_connection = res_connection
        self.node_attention = _HeadwiseQKV(
            hidden_dim,
            attention_hidden_dim,
            num_heads,
            depth,
            dropout,
            dynamic_depth=dynamic_depth,
            dense=dense,
            # Official GROVER's atom-message Head is constructed with
            # attach_fea=False; atom messages use the atom neighborhood and
            # do not inject the current bond-view state into Q/K/V.
            edge_dim=None,
        )
        self.edge_attention = _HeadwiseQKV(
            hidden_dim,
            attention_hidden_dim,
            num_heads,
            depth,
            dropout,
            dynamic_depth=dynamic_depth,
            dense=dense,
            edge_dim=None,
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atom_state: Tensor,
        bond_state: Tensor,
        node_owner: Tensor,
        node_neighbor: Tensor,
        node_edge_ids: Tensor,
        edge_owner: Tensor,
        edge_neighbor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        node_update = self.node_attention(
            atom_state,
            node_owner,
            node_neighbor,
            node_edge_ids,
            bond_state,
        )
        atom_state = atom_state + node_update if self.res_connection else node_update
        atom_state = self.node_norm(atom_state)

        if bond_state.shape[0]:
            edge_update = self.edge_attention(
                bond_state,
                edge_owner,
                edge_neighbor,
                None,
                None,
            )
            bond_state = bond_state + edge_update if self.res_connection else edge_update
            bond_state = self.edge_norm(bond_state)
        return atom_state, bond_state


class _StreamFFN(nn.Module):
    """Long-range initial-feature fusion used for one atom stream."""

    def __init__(self, atom_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + atom_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: Tensor, initial_atoms: Tensor) -> Tensor:
        return self.norm(self.network(torch.cat((state, initial_atoms), dim=-1)))


class _Readout(nn.Module):
    def __init__(self, hidden_dim: int, readout: str, attention_hidden_dim: int) -> None:
        super().__init__()
        self.readout = readout
        if readout == "self_attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1),
            )

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[0] == 0:
            return values.new_zeros(values.shape[1])
        if self.readout == "mean":
            return values.mean(dim=0)
        scores = self.attention(values).squeeze(-1)
        return (torch.softmax(scores, dim=0).unsqueeze(-1) * values).sum(dim=0)


class _PredictionFFN(nn.Module):
    def __init__(
        self, hidden_dim: int, num_targets: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(max(0, num_layers - 1)):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
        layers.append(nn.Linear(hidden_dim, num_targets))
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class GROVER(BaseMolecularModel):
    """GROVER's dual-view GTransformer property encoder.

    The official atom recipe is not identical to this project's canonical
    schema.  ``grover_f_atoms`` therefore records an explicit identity
    adaptation of canonical ``x`` and ``grover_f_bonds`` records source atom
    features concatenated with canonical bond features.  This is intentionally
    not a released-checkpoint compatibility claim.

    The shared trainer expects one tensor prediction.  The two official atom
    branches are therefore averaged by this adapter in both modes; the
    pretraining heads and the official branch-disagreement loss remain outside
    this first property-only port.
    """

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "grover_f_atoms",
        "grover_f_bonds",
        "grover_reverse_bond",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int = 1,
        hidden_dim: int = 128,
        num_mt_blocks: int = 2,
        num_heads: int = 4,
        attention_hidden_dim: int | None = None,
        depth: int = 3,
        ffn_num_layers: int = 2,
        dropout: float = 0.0,
        dynamic_depth: str = "truncnorm",
        dense: bool = False,
        res_connection: bool = False,
        readout: str = "mean",
        readout_attention_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_mt_blocks, "num_mt_blocks"),
            (num_heads, "num_heads"),
            (depth, "depth"),
            (ffn_num_layers, "ffn_num_layers"),
        ):
            _positive_int(value, name)
        _dropout(dropout)
        if dynamic_depth not in {"truncnorm", "none"}:
            raise ValueError("dynamic_depth must be 'truncnorm' or 'none'")
        if not isinstance(dense, bool) or not isinstance(res_connection, bool):
            raise ValueError("dense and res_connection must be booleans")
        if readout not in {"mean", "self_attention"}:
            raise ValueError("readout must be 'mean' or 'self_attention'")
        attention_hidden_dim = hidden_dim if attention_hidden_dim is None else attention_hidden_dim
        readout_attention_hidden_dim = (
            hidden_dim
            if readout_attention_hidden_dim is None
            else readout_attention_hidden_dim
        )
        _positive_int(attention_hidden_dim, "attention_hidden_dim")
        _positive_int(readout_attention_hidden_dim, "readout_attention_hidden_dim")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.feature_adaptation = ADAPTATION_NAME
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.depth = depth
        self.dynamic_depth = dynamic_depth
        self.dense = dense
        self.res_connection = res_connection
        self.readout_type = readout
        self.atom_input = nn.Linear(atom_dim, hidden_dim)
        self.bond_input = nn.Linear(atom_dim + bond_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _MTBlock(
                hidden_dim,
                attention_hidden_dim,
                num_heads,
                depth,
                dropout,
                dynamic_depth=dynamic_depth,
                dense=dense,
                res_connection=res_connection,
            )
            for _ in range(num_mt_blocks)
        )
        self.atom_from_atom = _StreamFFN(atom_dim, hidden_dim, dropout)
        self.atom_from_bond = _StreamFFN(atom_dim, hidden_dim, dropout)
        self.atom_readout = _Readout(hidden_dim, readout, readout_attention_hidden_dim)
        self.predictor_atom = _PredictionFFN(
            hidden_dim, num_targets, ffn_num_layers, dropout
        )
        self.predictor_bond = _PredictionFFN(
            hidden_dim, num_targets, ffn_num_layers, dropout
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return the shared-trainer-compatible mean of the two atom views."""
        atom_prediction, bond_prediction = self.forward_branches(batch)
        return (atom_prediction + bond_prediction) / 2

    def forward_branches(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Return official-style atom-view and bond-view predictions.

        The shared trainer currently requires one tensor, so :meth:`forward`
        averages these branches.  Keeping the branches available makes the
        official disagreement objective implementable by a future
        GROVER-specific task adapter without changing the encoder again.
        """
        tensors = self._batch_tensors(batch)
        (
            f_atoms,
            f_bonds,
            edge_index,
            reverse_bond,
            graph_batch,
            num_graphs,
        ) = tensors
        atom_predictions: list[Tensor] = []
        bond_predictions: list[Tensor] = []
        for graph_index in range(num_graphs):
            node_ids = torch.nonzero(graph_batch == graph_index, as_tuple=False).flatten()
            edge_mask = graph_batch[edge_index[0]] == graph_index
            edge_ids = torch.nonzero(edge_mask, as_tuple=False).flatten()
            local_edge_index = _local_edge_index(edge_index[:, edge_ids], node_ids)
            local_reverse = reverse_bond[edge_ids]
            atom_from_atom, atom_from_bond = self._encode_graph(
                f_atoms[node_ids],
                f_bonds[edge_ids],
                local_edge_index,
                local_reverse,
            )
            atom_predictions.append(self.predictor_atom(self._readout(atom_from_atom)))
            bond_predictions.append(self.predictor_bond(self._readout(atom_from_bond)))
        return torch.stack(atom_predictions, dim=0), torch.stack(bond_predictions, dim=0)

    def _encode_graph(
        self,
        f_atoms: Tensor,
        f_bonds: Tensor,
        edge_index: Tensor,
        reverse_bond: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        edge_count = edge_index.shape[1]
        node_owner = target
        node_neighbor = source
        node_edge_ids = torch.arange(edge_count, device=edge_index.device)
        edge_owner_list: list[int] = []
        edge_neighbor_list: list[int] = []
        incoming_by_node: list[list[int]] = [[] for _ in range(f_atoms.shape[0])]
        for edge_id, node in enumerate(target.tolist()):
            incoming_by_node[node].append(edge_id)
        for edge_id, source_node in enumerate(source.tolist()):
            for neighbor_edge in incoming_by_node[source_node]:
                if neighbor_edge != int(reverse_bond[edge_id].item()):
                    edge_owner_list.append(edge_id)
                    edge_neighbor_list.append(neighbor_edge)
        edge_owner = torch.tensor(edge_owner_list, dtype=torch.long, device=edge_index.device)
        edge_neighbor = torch.tensor(
            edge_neighbor_list, dtype=torch.long, device=edge_index.device
        )

        initial_atom_state = self.atom_input(f_atoms)
        atom_state = initial_atom_state
        bond_state = self.bond_input(f_bonds)
        for block in self.blocks:
            atom_state, bond_state = block(
                atom_state,
                bond_state,
                node_owner,
                node_neighbor,
                node_edge_ids,
                edge_owner,
                edge_neighbor,
            )

        incoming_bonds = atom_state.new_zeros(atom_state.shape)
        if edge_count:
            incoming_bonds.index_add_(0, target, bond_state)
        atom_from_atom = self.atom_from_atom(atom_state, f_atoms)
        atom_from_bond = self.atom_from_bond(incoming_bonds, f_atoms)
        return atom_from_atom, atom_from_bond

    def _readout(self, values: Tensor) -> Tensor:
        return self.atom_readout(values)

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, f_atoms, f_bonds, reverse_bond, graph_batch = values
        edge_count = edge_index.shape[1] if edge_index.ndim == 2 else -1
        if x.ndim != 2 or x.shape[0] < 1 or x.dtype != torch.float32:
            raise ValueError("batch.x must be a non-empty float32 tensor")
        if edge_index.ndim != 2 or edge_index.shape != (2, edge_count) or edge_index.dtype != torch.long:
            raise ValueError("batch.edge_index must have shape [2, E] and dtype torch.long")
        if edge_attr.shape != (edge_count, self.bond_dim) or edge_attr.dtype != torch.float32:
            raise ValueError(f"batch.edge_attr must have shape [E, {self.bond_dim}]")
        if f_atoms.shape != (x.shape[0], self.atom_dim) or f_atoms.dtype != torch.float32:
            raise ValueError("batch.grover_f_atoms has an invalid shape or dtype")
        if f_bonds.shape != (edge_count, self.atom_dim + self.bond_dim) or f_bonds.dtype != torch.float32:
            raise ValueError("batch.grover_f_bonds has an invalid shape or dtype")
        if reverse_bond.shape != (edge_count,) or reverse_bond.dtype != torch.long:
            raise ValueError("batch.grover_reverse_bond must have shape [E] and dtype torch.long")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=True,
        )
        # ``grover_reverse_bond`` intentionally remains local per molecule;
        # unlike ``edge_index`` it must not be offset by PyG's node-index
        # batching rule.  Validate each graph in its own edge coordinate
        # system, then convert to local indices in ``forward_branches``.
        edge_graph = graph_batch[edge_index[0]] if edge_count else graph_batch.new_empty(0)
        for graph_index in range(num_graphs):
            graph_edge_ids = torch.nonzero(
                edge_graph == graph_index, as_tuple=False
            ).flatten()
            if not graph_edge_ids.numel():
                continue
            graph_reverse = reverse_bond[graph_edge_ids]
            graph_edge_count = graph_edge_ids.numel()
            if graph_reverse.min() < 0 or graph_reverse.max() >= graph_edge_count:
                raise ValueError("batch.grover_reverse_bond contains an invalid edge index")
            if not torch.equal(
                graph_reverse[graph_reverse],
                torch.arange(graph_edge_count, device=graph_reverse.device),
            ):
                raise ValueError("batch.grover_reverse_bond must be an involution")
            graph_edges = edge_index[:, graph_edge_ids]
            if not torch.equal(graph_edges[:, graph_reverse], graph_edges.flip(0)):
                raise ValueError(
                    "batch.grover_reverse_bond must map each edge to its reverse"
                )
        if any(value.device != x.device for value in values):
            raise ValueError("GROVER batch tensors must share the node device")
        return f_atoms, f_bonds, edge_index, reverse_bond, graph_batch, num_graphs


def _local_edge_index(edge_index: Tensor, node_ids: Tensor) -> Tensor:
    if edge_index.shape[1] == 0:
        return edge_index
    local_map = torch.full(
        (int(node_ids.max().item()) + 1,), -1, dtype=torch.long, device=node_ids.device
    )
    local_map[node_ids] = torch.arange(node_ids.shape[0], device=node_ids.device)
    return local_map[edge_index]


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


__all__ = ["GROVER"]
