"""Graphormer-style Transformer-M architecture over packed PyG graph data."""

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
    BOND_TYPE_COUNT,
    UNREACHABLE_SPD,
)


class _TransformerMEncoderLayer(nn.Module):
    """Pre-norm multi-head self-attention with an additive score bias."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, attention_bias: Tensor) -> Tensor:
        residual = values
        normalized = self.norm1(values)
        qkv = self.qkv(normalized).reshape(
            values.shape[0], values.shape[1], 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + attention_bias.unsqueeze(0)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.permute(0, 2, 1, 3).reshape_as(values)
        values = residual + self.dropout(self.out_proj(attended))
        return values + self.dropout(self.ffn(self.norm2(values)))


class TransformerM(BaseMolecularModel):
    """Transformer-M with explicit 2D topology and optional 3D geometry.

    The model consumes categorical atom features derived by the
    ``transformer_m_inputs`` transform.  Shortest-path and multi-hop edge
    records are packed sparsely in the PyG batch and rebuilt into one dense
    attention-bias matrix per molecule.  This keeps model-specific dense
    batching out of the shared data container.

    The scalar prediction head reads the virtual graph token.  The optional
    denoising task from the pretraining repository is intentionally outside
    the shared single-output task contract.
    """

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "transformer_m_x",
        "transformer_m_in_degree",
        "transformer_m_out_degree",
        "transformer_m_pair_index",
        "transformer_m_spatial_pos",
        "transformer_m_path_index",
        "transformer_m_path_step",
        "transformer_m_path_type",
        "batch",
    )

    def __init__(
        self,
        num_targets: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_degree: int = 512,
        max_spatial_pos: int = UNREACHABLE_SPD,
        max_path_length: int = UNREACHABLE_SPD,
        num_gaussians: int = 128,
        atom_feature_vocab_sizes: Sequence[int] = ATOM_FEATURE_VOCAB_SIZES,
        mode_prob: Sequence[float] = (0.2, 0.2, 0.6),
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (ffn_dim, "ffn_dim"),
            (max_degree, "max_degree"),
            (max_spatial_pos, "max_spatial_pos"),
            (max_path_length, "max_path_length"),
            (num_gaussians, "num_gaussians"),
        ):
            _positive_int(value, name)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        _dropout(dropout)
        vocab_sizes = _positive_ints(atom_feature_vocab_sizes, "atom_feature_vocab_sizes")
        if len(vocab_sizes) != ATOM_FEATURE_COUNT:
            raise ValueError(
                f"atom_feature_vocab_sizes must contain {ATOM_FEATURE_COUNT} entries"
            )
        if tuple(vocab_sizes) != ATOM_FEATURE_VOCAB_SIZES:
            raise ValueError(
                "Transformer-M currently requires the canonical categorical feature vocabulary"
            )
        probabilities = _probabilities(mode_prob)
        if max_spatial_pos > UNREACHABLE_SPD:
            raise ValueError("max_spatial_pos must not exceed the audit sentinel")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_degree = max_degree
        self.max_spatial_pos = max_spatial_pos
        self.max_path_length = max_path_length
        self.num_gaussians = num_gaussians
        self.mode_prob = probabilities

        self.atom_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_dim) for size in vocab_sizes
        )
        self.in_degree_embedding = nn.Embedding(max_degree, hidden_dim)
        self.out_degree_embedding = nn.Embedding(max_degree, hidden_dim)
        self.graph_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.virtual_token_bias = nn.Parameter(torch.zeros(()))

        self.spatial_pos_embedding = nn.Embedding(UNREACHABLE_SPD + 1, num_heads)
        self.path_type_embedding = nn.Embedding(BOND_TYPE_COUNT + 1, num_heads, padding_idx=0)
        self.path_step_embedding = nn.Embedding(max_path_length, num_heads)

        self.atom_type_vocab_size = vocab_sizes[0]
        self.gaussian_centers = nn.Parameter(torch.linspace(0.0, 10.0, num_gaussians))
        self.gaussian_widths = nn.Parameter(torch.full((num_gaussians,), 0.5))
        self.atom_pair_embedding = nn.Embedding(
            self.atom_type_vocab_size * self.atom_type_vocab_size,
            num_gaussians,
        )
        self.three_d_bias_projection = nn.Linear(num_gaussians, num_heads, bias=False)
        self.three_d_node_projection = nn.Linear(num_gaussians, hidden_dim, bias=False)

        self.layers = nn.ModuleList(
            _TransformerMEncoderLayer(hidden_dim, num_heads, ffn_dim, float(dropout))
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
        tensors = self._batch_tensors(batch)
        (
            x,
            _edge_index,
            edge_attr,
            categorical_x,
            in_degree,
            out_degree,
            pair_index,
            spatial_pos,
            path_index,
            path_step,
            path_type,
            graph_batch,
            pos,
            num_graphs,
        ) = tensors
        del edge_attr
        outputs: list[Tensor] = []
        for graph_index in range(num_graphs):
            node_ids = torch.nonzero(graph_batch == graph_index, as_tuple=False).flatten()
            outputs.append(
                self._forward_graph(
                    x[node_ids],
                    categorical_x[node_ids],
                    in_degree[node_ids],
                    out_degree[node_ids],
                    pair_index,
                    spatial_pos,
                    path_index,
                    path_step,
                    path_type,
                    node_ids,
                    None if pos is None else pos[node_ids],
                )
            )
        return self.predictor(torch.cat(outputs, dim=0))

    def _forward_graph(
        self,
        x: Tensor,
        categorical_x: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        pair_index: Tensor,
        spatial_pos: Tensor,
        path_index: Tensor,
        path_step: Tensor,
        path_type: Tensor,
        node_ids: Tensor,
        pos: Tensor | None,
    ) -> Tensor:
        node_count = x.shape[0]
        local_map = torch.full(
            (int(node_ids.max().item()) + 1,), -1, dtype=torch.long, device=node_ids.device
        )
        local_map[node_ids] = torch.arange(node_count, device=node_ids.device)
        pair_mask = torch.isin(pair_index[0], node_ids) & torch.isin(
            pair_index[1], node_ids
        )
        local_pair = local_map[pair_index[:, pair_mask]]
        pair_values = spatial_pos[pair_mask]
        pair_ids = local_pair[0] * node_count + local_pair[1]
        if torch.unique(pair_ids).numel() != node_count * node_count:
            raise ValueError("batch.transformer_m_pair_index must enumerate every ordered pair")

        use_2d, use_3d = self._channel_modes(pos is not None)
        node_values = x.new_zeros((node_count, self.hidden_dim))
        for column, embedding in enumerate(self.atom_embeddings):
            node_values = node_values + embedding(categorical_x[:, column])
        if use_2d:
            node_values = node_values + self.in_degree_embedding(in_degree)
            node_values = node_values + self.out_degree_embedding(out_degree)

        attention_bias = x.new_zeros((self.num_heads, node_count + 1, node_count + 1))
        atom_bias = x.new_zeros((node_count * node_count, self.num_heads))
        if use_2d:
            spd = self.spatial_pos_embedding(pair_values).to(dtype=x.dtype)
            atom_bias = atom_bias + spd
            path_mask = torch.isin(path_index[0], node_ids) & torch.isin(
                path_index[1], node_ids
            )
            if bool(path_mask.any()):
                local_path = local_map[path_index[:, path_mask]]
                path_ids = local_path[0] * node_count + local_path[1]
                path_steps = path_step[path_mask].clamp_max(self.max_path_length - 1)
                path_values = self.path_type_embedding(path_type[path_mask])
                path_values = path_values * self.path_step_embedding(path_steps)
                path_bias = torch.zeros_like(atom_bias)
                path_bias.index_add_(0, path_ids, path_values)
                path_counts = torch.zeros(
                    node_count * node_count, dtype=x.dtype, device=x.device
                )
                path_counts.index_add_(
                    0, path_ids, torch.ones(path_ids.shape[0], dtype=x.dtype, device=x.device)
                )
                atom_bias = atom_bias + path_bias / path_counts.clamp_min(1).unsqueeze(-1)

        if use_3d:
            assert pos is not None
            distances = torch.linalg.vector_norm(
                pos[local_pair[0]] - pos[local_pair[1]], dim=-1
            )
            rbf = self._gaussian_features(distances)
            atom_types = categorical_x[:, 0]
            pair_types = atom_types[local_pair[0]] * self.atom_type_vocab_size + atom_types[
                local_pair[1]
            ]
            conditioned = rbf * self.atom_pair_embedding(pair_types)
            atom_bias = atom_bias + self.three_d_bias_projection(conditioned)
            node_rbf = torch.zeros(
                node_count, self.num_gaussians, dtype=x.dtype, device=x.device
            )
            node_rbf.index_add_(0, local_pair[1], rbf)
            node_values = node_values + 0.01 * self.three_d_node_projection(node_rbf)

        atom_bias = atom_bias.reshape(node_count, node_count, self.num_heads).permute(2, 0, 1)
        attention_bias[:, 1:, 1:] = atom_bias
        attention_bias[:, 0, 1:] = self.virtual_token_bias
        attention_bias[:, 1:, 0] = self.virtual_token_bias

        if use_2d:
            unreachable = pair_values >= self.max_spatial_pos
            if bool(unreachable.any()):
                unreachable_ids = pair_ids[unreachable]
                local_source = unreachable_ids // node_count
                local_target = unreachable_ids % node_count
                attention_bias[:, local_source + 1, local_target + 1] = -torch.inf

        values = torch.cat((self.graph_token.to(dtype=x.dtype), node_values), dim=0).unsqueeze(0)
        for layer in self.layers:
            values = layer(values, attention_bias)
        return values[:, 0, :]

    def _gaussian_features(self, distances: Tensor) -> Tensor:
        widths = self.gaussian_widths.abs().clamp_min(1e-4)
        return torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.gaussian_centers) / widths) ** 2)

    def _channel_modes(self, has_3d: bool) -> tuple[bool, bool]:
        if not has_3d:
            return True, False
        if not self.training:
            return True, True
        draw = float(torch.rand((), device=self.graph_token.device).item())
        both, two_d, _three_d = self.mode_prob
        if draw < both:
            return True, True
        if draw < both + two_d:
            return True, False
        return False, True

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, int]:
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
            path_type,
            graph_batch,
        ) = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(categorical_x, Tensor)
        assert isinstance(in_degree, Tensor)
        assert isinstance(out_degree, Tensor)
        assert isinstance(pair_index, Tensor)
        assert isinstance(spatial_pos, Tensor)
        assert isinstance(path_index, Tensor)
        assert isinstance(path_step, Tensor)
        assert isinstance(path_type, Tensor)
        assert isinstance(graph_batch, Tensor)
        if x.ndim != 2 or x.shape[0] < 1 or x.dtype != torch.float32:
            raise ValueError("batch.x must be a non-empty float32 tensor")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("batch.edge_attr must align with batch.edge_index")
        if edge_attr.dtype != torch.float32:
            raise ValueError("batch.edge_attr must have dtype torch.float32")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=True,
        )
        if categorical_x.shape != (x.shape[0], ATOM_FEATURE_COUNT) or categorical_x.dtype != torch.long:
            raise ValueError("batch.transformer_m_x must have shape [N, 8] and dtype torch.long")
        for column, size in enumerate(ATOM_FEATURE_VOCAB_SIZES):
            if categorical_x[:, column].min() < 0 or categorical_x[:, column].max() >= size:
                raise ValueError("batch.transformer_m_x contains an invalid categorical id")
        for degree, name in ((in_degree, "in_degree"), (out_degree, "out_degree")):
            if degree.shape != (x.shape[0],) or degree.dtype != torch.long:
                raise ValueError(f"batch.transformer_m_{name} must have shape [N] and dtype torch.long")
            if degree.numel() and (degree.min() < 0 or degree.max() >= self.max_degree):
                raise ValueError(f"batch.transformer_m_{name} exceeds max_degree")
        _validate_pairs(pair_index, spatial_pos, graph_batch, num_graphs, x.shape[0])
        _validate_paths(path_index, path_step, path_type, graph_batch, x.shape[0])
        if any(value.device != x.device for value in values):
            raise ValueError("Transformer-M batch tensors must share the node device")
        pos = getattr(batch, "pos", None)
        if pos is not None and (
            not isinstance(pos, Tensor)
            or pos.shape != (x.shape[0], 3)
            or pos.dtype != torch.float32
            or not torch.isfinite(pos).all()
        ):
            raise ValueError("batch.pos must have shape [N, 3] and finite float32 values")
        return (
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
            path_type,
            graph_batch,
            pos,
            num_graphs,
        )


def _validate_pairs(
    pair_index: Tensor,
    spatial_pos: Tensor,
    graph_batch: Tensor,
    num_graphs: int,
    num_nodes: int,
) -> None:
    if pair_index.ndim != 2 or pair_index.shape[0] != 2 or pair_index.dtype != torch.long:
        raise ValueError("batch.transformer_m_pair_index must have shape [2, P] and dtype torch.long")
    if spatial_pos.shape != (pair_index.shape[1],) or spatial_pos.dtype != torch.long:
        raise ValueError("batch.transformer_m_spatial_pos must align with pair_index")
    if pair_index.numel() == 0 or pair_index.min() < 0 or pair_index.max() >= num_nodes:
        raise ValueError("batch.transformer_m_pair_index contains an invalid node index")
    if spatial_pos.min() < 1 or spatial_pos.max() > UNREACHABLE_SPD:
        raise ValueError("batch.transformer_m_spatial_pos contains an invalid SPD value")
    source, target = pair_index
    if not torch.equal(graph_batch[source], graph_batch[target]):
        raise ValueError("batch.transformer_m_pair_index must not connect different graphs")
    counts = torch.bincount(graph_batch, minlength=num_graphs)
    if pair_index.shape[1] != int(counts.square().sum().item()):
        raise ValueError("batch.transformer_m_pair_index must enumerate every ordered pair")
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    for graph_index, (start, stop) in enumerate(
        zip(starts[:-1].tolist(), starts[1:].tolist(), strict=True)
    ):
        graph_mask = graph_batch[source] == graph_index
        local_source = source[graph_mask] - start
        local_target = target[graph_mask] - start
        pair_ids = local_source * (stop - start) + local_target
        if torch.unique(pair_ids).numel() != (stop - start) ** 2:
            raise ValueError("batch.transformer_m_pair_index must not omit or duplicate pairs")
def _validate_paths(
    path_index: Tensor,
    path_step: Tensor,
    path_type: Tensor,
    graph_batch: Tensor,
    num_nodes: int,
) -> None:
    if path_index.ndim != 2 or path_index.shape[0] != 2 or path_index.dtype != torch.long:
        raise ValueError("batch.transformer_m_path_index must have shape [2, K] and dtype torch.long")
    if path_step.shape != (path_index.shape[1],) or path_step.dtype != torch.long:
        raise ValueError("batch.transformer_m_path_step must align with path_index")
    if path_type.shape != (path_index.shape[1],) or path_type.dtype != torch.long:
        raise ValueError("batch.transformer_m_path_type must align with path_index")
    if path_index.numel():
        if path_index.min() < 0 or path_index.max() >= num_nodes:
            raise ValueError("batch.transformer_m_path_index contains an invalid node index")
        if not torch.equal(graph_batch[path_index[0]], graph_batch[path_index[1]]):
            raise ValueError("batch.transformer_m_path_index must not connect different graphs")
        if path_step.min() < 0 or path_type.min() < 1 or path_type.max() > BOND_TYPE_COUNT:
            raise ValueError("Transformer-M path records contain invalid values")


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


def _probabilities(value: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("mode_prob must contain three probabilities")
    probabilities = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0 for item in probabilities):
        raise ValueError("mode_prob must contain finite non-negative probabilities")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
        raise ValueError("mode_prob must sum to 1")
    return probabilities


__all__ = ["TransformerM"]
