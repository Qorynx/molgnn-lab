"""Dense 2D Graphormer over packed PyG molecular graphs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    ATOM_FEATURE_COUNT,
    ATOM_FEATURE_VOCAB_SIZES,
    BOND_FEATURE_COUNT,
    BOND_FEATURE_VOCAB_SIZES,
    FEATURE_EMBEDDING_OFFSET,
    UNREACHABLE_SPD,
)


class _GraphormerEncoderLayer(nn.Module):
    """One Graphormer attention/FFN block with selectable LN placement."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        *,
        pre_layernorm: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.pre_layernorm = pre_layernorm
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attention_out = nn.Linear(hidden_dim, hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, attention_bias: Tensor) -> Tensor:
        residual = values
        attention_input = self.attention_norm(values) if self.pre_layernorm else values
        qkv = self.qkv(attention_input).reshape(
            values.shape[0], values.shape[1], 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores + attention_bias.unsqueeze(0), dim=-1)
        attended = torch.matmul(weights, value).permute(0, 2, 1, 3).reshape_as(values)
        values = residual + self.dropout(self.attention_out(attended))
        if not self.pre_layernorm:
            values = self.attention_norm(values)

        residual = values
        ffn_input = self.ffn_norm(values) if self.pre_layernorm else values
        values = residual + self.dropout(self.ffn(ffn_input))
        return values if self.pre_layernorm else self.ffn_norm(values)


class _GraphormerAttentionBias(nn.Module):
    """Shared SPD and multi-hop edge attention bias used by every layer."""

    def __init__(self, num_heads: int, max_path_length: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_path_length = max_path_length
        self.spatial_pos_embedding = nn.Embedding(UNREACHABLE_SPD + 1, num_heads)
        self.graph_token_virtual_distance = nn.Parameter(torch.zeros(num_heads))
        self.edge_encoder = nn.Embedding(
            BOND_FEATURE_COUNT * FEATURE_EMBEDDING_OFFSET + 1,
            num_heads,
        )
        self.edge_dis_encoder = nn.Parameter(
            torch.empty(max_path_length, num_heads, num_heads)
        )
        nn.init.xavier_uniform_(self.edge_dis_encoder)

    def forward(
        self,
        pair_ids: Tensor,
        spatial_pos: Tensor,
        path_ids: Tensor,
        path_step: Tensor,
        path_edge_type: Tensor,
        node_count: int,
    ) -> Tensor:
        atom_bias = torch.zeros(
            node_count * node_count,
            self.num_heads,
            dtype=self.spatial_pos_embedding.weight.dtype,
            device=spatial_pos.device,
        )
        atom_bias[pair_ids] = self.spatial_pos_embedding(spatial_pos)

        if path_ids.numel():
            edge_values = self.edge_encoder(_single_embedding_ids(path_edge_type)).mean(dim=1)
            edge_values = torch.bmm(
                edge_values.unsqueeze(1), self.edge_dis_encoder[path_step]
            ).squeeze(1)
            path_bias = torch.zeros_like(atom_bias)
            path_bias.index_add_(0, path_ids, edge_values)
            path_lengths = spatial_pos.new_zeros(node_count * node_count)
            path_lengths[pair_ids] = spatial_pos
            atom_bias = atom_bias + path_bias / path_lengths.clamp_min(1).unsqueeze(-1)

        attention_bias = atom_bias.reshape(
            node_count, node_count, self.num_heads
        ).permute(2, 0, 1).new_zeros((self.num_heads, node_count + 1, node_count + 1))
        attention_bias[:, 1:, 1:] = atom_bias.reshape(
            node_count, node_count, self.num_heads
        ).permute(2, 0, 1)
        attention_bias[:, 0, 1:] = self.graph_token_virtual_distance.unsqueeze(-1)
        attention_bias[:, 1:, 0] = self.graph_token_virtual_distance.unsqueeze(-1)
        return attention_bias


class Graphormer(BaseMolecularModel):
    """Official-style 2D Graphormer with packed structural inputs.

    The ``graphormer_inputs`` transform prepares categorical atom and bond ids,
    all shortest-path pairs, and one edge-feature sequence per shortest path.
    Dense attention is rebuilt per graph here, leaving shared PyG data/training
    code unchanged.  This runtime deliberately excludes Graphormer3D.
    """

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "graphormer_x",
        "graphormer_in_degree",
        "graphormer_out_degree",
        "graphormer_pair_index",
        "graphormer_spatial_pos",
        "graphormer_path_index",
        "graphormer_path_step",
        "graphormer_path_edge_type",
        "batch",
    )

    def __init__(
        self,
        num_targets: int = 1,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 32,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_degree: int = 512,
        max_path_length: int = UNREACHABLE_SPD,
        spatial_pos_max: int = 1024,
        pre_layernorm: bool = False,
        embedding_layernorm: bool = True,
        atom_feature_vocab_sizes: Sequence[int] = ATOM_FEATURE_VOCAB_SIZES,
        bond_feature_vocab_sizes: Sequence[int] = BOND_FEATURE_VOCAB_SIZES,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (max_degree, "max_degree"),
            (max_path_length, "max_path_length"),
            (spatial_pos_max, "spatial_pos_max"),
        ):
            _positive_int(value, name)
        if ffn_dim is not None:
            _positive_int(ffn_dim, "ffn_dim")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        _dropout(dropout)
        if not isinstance(pre_layernorm, bool) or not isinstance(embedding_layernorm, bool):
            raise ValueError("pre_layernorm and embedding_layernorm must be booleans")
        if tuple(_positive_ints(atom_feature_vocab_sizes, "atom_feature_vocab_sizes")) != ATOM_FEATURE_VOCAB_SIZES:
            raise ValueError("Graphormer requires the canonical categorical atom feature vocabulary")
        if tuple(_positive_ints(bond_feature_vocab_sizes, "bond_feature_vocab_sizes")) != BOND_FEATURE_VOCAB_SIZES:
            raise ValueError("Graphormer requires the canonical categorical bond feature vocabulary")

        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.max_path_length = max_path_length
        self.spatial_pos_max = spatial_pos_max
        self.atom_encoder = nn.Embedding(
            ATOM_FEATURE_COUNT * FEATURE_EMBEDDING_OFFSET + 1, hidden_dim
        )
        self.in_degree_encoder = nn.Embedding(max_degree, hidden_dim)
        self.out_degree_encoder = nn.Embedding(max_degree, hidden_dim)
        self.graph_token = nn.Parameter(torch.empty(1, hidden_dim))
        self.embedding_norm = nn.LayerNorm(hidden_dim) if embedding_layernorm else nn.Identity()
        self.embedding_dropout = nn.Dropout(dropout)
        self.attention_bias = _GraphormerAttentionBias(num_heads, max_path_length)
        effective_ffn_dim = hidden_dim if ffn_dim is None else ffn_dim
        self.layers = nn.ModuleList(
            _GraphormerEncoderLayer(
                hidden_dim,
                num_heads,
                effective_ffn_dim,
                dropout,
                pre_layernorm=pre_layernorm,
            )
            for _ in range(num_layers)
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_targets),
        )
        nn.init.normal_(self.graph_token, mean=0.0, std=hidden_dim**-0.5)

    def forward(self, batch: Batch) -> Tensor:
        (
            x,
            categorical_x,
            in_degree,
            out_degree,
            pair_index,
            spatial_pos,
            path_index,
            path_step,
            path_edge_type,
            graph_batch,
            num_graphs,
        ) = self._batch_tensors(batch)
        outputs: list[Tensor] = []
        for graph_index in range(num_graphs):
            node_ids = torch.nonzero(graph_batch == graph_index, as_tuple=False).flatten()
            outputs.append(
                self._forward_graph(
                    x.dtype,
                    categorical_x[node_ids],
                    in_degree[node_ids],
                    out_degree[node_ids],
                    pair_index,
                    spatial_pos,
                    path_index,
                    path_step,
                    path_edge_type,
                    node_ids,
                )
            )
        return self.predictor(torch.cat(outputs, dim=0))

    def _forward_graph(
        self,
        dtype: torch.dtype,
        categorical_x: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        pair_index: Tensor,
        spatial_pos: Tensor,
        path_index: Tensor,
        path_step: Tensor,
        path_edge_type: Tensor,
        node_ids: Tensor,
    ) -> Tensor:
        node_count = categorical_x.shape[0]
        local_map = torch.full(
            (int(node_ids.max().item()) + 1,), -1, dtype=torch.long, device=node_ids.device
        )
        local_map[node_ids] = torch.arange(node_count, device=node_ids.device)
        pair_mask = torch.isin(pair_index[0], node_ids) & torch.isin(pair_index[1], node_ids)
        local_pair = local_map[pair_index[:, pair_mask]]
        pair_ids = local_pair[0] * node_count + local_pair[1]
        pair_values = spatial_pos[pair_mask]
        if torch.unique(pair_ids).numel() != node_count * node_count:
            raise ValueError("batch.graphormer_pair_index must enumerate every ordered pair")

        path_mask = torch.isin(path_index[0], node_ids) & torch.isin(path_index[1], node_ids)
        local_path = local_map[path_index[:, path_mask]]
        path_ids = local_path[0] * node_count + local_path[1]
        local_path_step = path_step[path_mask]
        local_path_edge_type = path_edge_type[path_mask]
        _validate_path_lengths(pair_ids, pair_values, path_ids, node_count)

        node_values = self.atom_encoder(_single_embedding_ids(categorical_x)).sum(dim=1)
        node_values = node_values + self.in_degree_encoder(in_degree)
        node_values = node_values + self.out_degree_encoder(out_degree)
        values = torch.cat((self.graph_token.to(dtype=dtype), node_values), dim=0).unsqueeze(0)
        values = self.embedding_dropout(self.embedding_norm(values))

        attention_bias = self.attention_bias(
            pair_ids,
            pair_values,
            path_ids,
            local_path_step,
            local_path_edge_type,
            node_count,
        ).to(dtype=dtype)
        far_pairs = pair_values >= self.spatial_pos_max
        if bool(far_pairs.any()):
            far_ids = pair_ids[far_pairs]
            attention_bias[:, far_ids // node_count + 1, far_ids % node_count + 1] = -torch.inf
        for layer in self.layers:
            values = layer(values, attention_bias)
        return values[:, 0, :]

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        (
            x,
            edge_index,
            edge_attr,
            categorical_x,
            in_degree,
            out_degree,
            pair_index,
            spatial_pos,
            path_index,
            path_step,
            path_edge_type,
            graph_batch,
        ) = values
        assert all(isinstance(value, Tensor) for value in values)
        if x.ndim != 2 or x.shape[0] < 1 or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must be a non-empty finite float32 tensor")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1] or edge_attr.dtype != torch.float32:
            raise ValueError("batch.edge_attr must be float32 and align with batch.edge_index")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=True,
        )
        if categorical_x.shape != (x.shape[0], ATOM_FEATURE_COUNT) or categorical_x.dtype != torch.long:
            raise ValueError("batch.graphormer_x must have shape [N, 8] and dtype torch.long")
        _validate_categorical(categorical_x, ATOM_FEATURE_VOCAB_SIZES, "graphormer_x")
        for degree, name in ((in_degree, "in_degree"), (out_degree, "out_degree")):
            if degree.shape != (x.shape[0],) or degree.dtype != torch.long:
                raise ValueError(f"batch.graphormer_{name} must have shape [N] and dtype torch.long")
            if degree.min() < 0 or degree.max() >= self.max_degree:
                raise ValueError(f"batch.graphormer_{name} exceeds max_degree")
        _validate_pairs(pair_index, spatial_pos, graph_batch, num_graphs, x.shape[0])
        _validate_paths(path_index, path_step, path_edge_type, graph_batch, x.shape[0], self.max_path_length)
        if any(value.device != x.device for value in values):
            raise ValueError("Graphormer batch tensors must share the node device")
        return (
            x,
            categorical_x,
            in_degree,
            out_degree,
            pair_index,
            spatial_pos,
            path_index,
            path_step,
            path_edge_type,
            graph_batch,
            num_graphs,
        )


def _single_embedding_ids(values: Tensor) -> Tensor:
    offsets = 1 + torch.arange(
        values.shape[1], device=values.device, dtype=torch.long
    ) * FEATURE_EMBEDDING_OFFSET
    return values + offsets


def _validate_categorical(values: Tensor, vocab_sizes: Sequence[int], name: str) -> None:
    for column, size in enumerate(vocab_sizes):
        if values[:, column].min() < 0 or values[:, column].max() >= size:
            raise ValueError(f"batch.{name} contains an invalid categorical id")


def _validate_pairs(
    pair_index: Tensor,
    spatial_pos: Tensor,
    graph_batch: Tensor,
    num_graphs: int,
    num_nodes: int,
) -> None:
    if pair_index.ndim != 2 or pair_index.shape[0] != 2 or pair_index.dtype != torch.long:
        raise ValueError("batch.graphormer_pair_index must have shape [2, P] and dtype torch.long")
    if spatial_pos.shape != (pair_index.shape[1],) or spatial_pos.dtype != torch.long:
        raise ValueError("batch.graphormer_spatial_pos must align with pair_index")
    if pair_index.numel() == 0 or pair_index.min() < 0 or pair_index.max() >= num_nodes:
        raise ValueError("batch.graphormer_pair_index contains an invalid node index")
    if spatial_pos.min() < 0 or spatial_pos.max() > UNREACHABLE_SPD:
        raise ValueError("batch.graphormer_spatial_pos contains an invalid SPD value")
    source, target = pair_index
    if not torch.equal(graph_batch[source], graph_batch[target]):
        raise ValueError("batch.graphormer_pair_index must not connect different graphs")
    counts = torch.bincount(graph_batch, minlength=num_graphs)
    if pair_index.shape[1] != int(counts.square().sum().item()):
        raise ValueError("batch.graphormer_pair_index must enumerate every ordered pair")
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    for graph_index, (start, stop) in enumerate(
        zip(starts[:-1].tolist(), starts[1:].tolist(), strict=True)
    ):
        graph_mask = graph_batch[source] == graph_index
        local_source = source[graph_mask] - start
        local_target = target[graph_mask] - start
        pair_ids = local_source * (stop - start) + local_target
        if torch.unique(pair_ids).numel() != (stop - start) ** 2:
            raise ValueError("batch.graphormer_pair_index must not omit or duplicate pairs")


def _validate_paths(
    path_index: Tensor,
    path_step: Tensor,
    path_edge_type: Tensor,
    graph_batch: Tensor,
    num_nodes: int,
    max_path_length: int,
) -> None:
    if path_index.ndim != 2 or path_index.shape[0] != 2 or path_index.dtype != torch.long:
        raise ValueError("batch.graphormer_path_index must have shape [2, K] and dtype torch.long")
    if path_step.shape != (path_index.shape[1],) or path_step.dtype != torch.long:
        raise ValueError("batch.graphormer_path_step must align with path_index")
    if path_edge_type.shape != (path_index.shape[1], BOND_FEATURE_COUNT) or path_edge_type.dtype != torch.long:
        raise ValueError("batch.graphormer_path_edge_type must have shape [K, 4] and dtype torch.long")
    if path_index.numel():
        if path_index.min() < 0 or path_index.max() >= num_nodes:
            raise ValueError("batch.graphormer_path_index contains an invalid node index")
        if not torch.equal(graph_batch[path_index[0]], graph_batch[path_index[1]]):
            raise ValueError("batch.graphormer_path_index must not connect different graphs")
        if path_step.min() < 0 or path_step.max() >= max_path_length:
            raise ValueError("batch.graphormer_path_step exceeds max_path_length")
        _validate_categorical(path_edge_type, BOND_FEATURE_VOCAB_SIZES, "graphormer_path_edge_type")


def _validate_path_lengths(
    pair_ids: Tensor, spatial_pos: Tensor, path_ids: Tensor, node_count: int
) -> None:
    expected = spatial_pos.new_zeros(node_count * node_count)
    expected[pair_ids] = torch.where(
        spatial_pos == UNREACHABLE_SPD,
        torch.zeros_like(spatial_pos),
        spatial_pos,
    )
    actual = torch.bincount(path_ids, minlength=node_count * node_count)
    if not torch.equal(actual, expected):
        raise ValueError("Graphormer path records must contain one edge per shortest-path hop")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_ints(value: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of positive integers")
    result = tuple(value)
    for item in result:
        _positive_int(item, name)
    return result


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


__all__ = ["Graphormer"]
