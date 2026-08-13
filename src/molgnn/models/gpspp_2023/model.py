"""The 2D GPS++ hybrid molecular graph architecture.

This module ports the reusable GPS++ trunk to the project's canonical sparse
PyG graph boundary.  It deliberately adapts the source PCQM4Mv2 categorical
encoder to continuous ``x`` and ``edge_attr`` projections, while preserving
the architecture that follows: directed edge/node/global message passing in
parallel with shortest-path-biased self-attention, then a residual FFN.

Coordinates, PCQM4Mv2-specific positional encodings, and auxiliary denoising
or reconstruction losses are intentionally outside this 2D runtime model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GPSPlusPlusBlock, LayerNormMLP


class GPSPlusPlus(BaseMolecularModel):
    """Hybrid local-MPNN and SPD-biased Transformer graph predictor.

    ``gpspp_inputs`` derives a complete, ordered atom-pair view before PyG
    batching.  ``gpspp_pair_index`` is used only to construct a graph-local
    shortest-path-distance bias for self-attention; the local branch still
    uses the canonical directed covalent ``edge_index`` and ``edge_attr``.

    The model accepts continuous atom and bond features so it remains usable
    with the shared canonical featurizer and alternative MoleculeNet datasets.
    It does not reinterpret those tensors as Graphcore's PCQM4Mv2 categorical
    feature profile.
    """

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "gpspp_pair_index",
        "gpspp_spd",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        node_dim: int = 256,
        edge_dim: int = 128,
        global_dim: int = 64,
        depth: int = 16,
        num_heads: int = 32,
        expansion: int = 4,
        max_spd: int = 100,
        encoder_dropout: float = 0.18,
        node_dropout: float = 0.3,
        edge_dropout: float = 0.0035,
        global_dropout: float = 0.35,
        attention_dropout: float = 0.3,
        ffn_dropout: float = 0.0,
        max_stochastic_depth: float = 0.3,
        decoder_hidden_dim: int = 256,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (node_dim, "node_dim"),
            (edge_dim, "edge_dim"),
            (global_dim, "global_dim"),
            (depth, "depth"),
            (num_heads, "num_heads"),
            (expansion, "expansion"),
            (max_spd, "max_spd"),
            (decoder_hidden_dim, "decoder_hidden_dim"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        if node_dim % num_heads:
            raise ValueError("node_dim must be divisible by num_heads")
        for value, name in (
            (encoder_dropout, "encoder_dropout"),
            (node_dropout, "node_dropout"),
            (edge_dropout, "edge_dropout"),
            (global_dropout, "global_dropout"),
            (attention_dropout, "attention_dropout"),
            (ffn_dropout, "ffn_dropout"),
            (max_stochastic_depth, "max_stochastic_depth"),
        ):
            _dropout(value, name)

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.depth = depth
        self.num_heads = num_heads
        self.expansion = expansion
        self.max_spd = max_spd
        self.num_targets = num_targets
        self.max_stochastic_depth = float(max_stochastic_depth)

        self.node_encoder = LayerNormMLP(
            atom_dim,
            node_dim,
            hidden_dim=expansion * node_dim,
            dropout=encoder_dropout,
        )
        self.edge_encoder = LayerNormMLP(
            bond_dim,
            edge_dim,
            hidden_dim=expansion * edge_dim,
            dropout=encoder_dropout,
        )
        self.global_embedding = nn.Parameter(torch.empty(global_dim))
        nn.init.normal_(self.global_embedding, mean=0.0, std=global_dim**-0.5)

        self.blocks = nn.ModuleList(
            GPSPlusPlusBlock(
                node_dim,
                edge_dim,
                global_dim,
                num_heads,
                max_spd,
                node_hidden_dim=expansion * node_dim,
                edge_hidden_dim=expansion * edge_dim,
                global_hidden_dim=expansion * global_dim,
                ffn_hidden_dim=expansion * node_dim,
                node_dropout=node_dropout,
                edge_dropout=edge_dropout,
                global_dropout=global_dropout,
                ffn_dropout=ffn_dropout,
                attention_dropout=attention_dropout,
                graph_dropout=_drop_path_rate(
                    block_index, depth, float(max_stochastic_depth)
                ),
                # The published profile disables local-MPNN stochastic depth.
                local_graph_dropout=0.0,
            )
            for block_index in range(depth)
        )
        self.decoder = nn.Sequential(
            nn.Linear(node_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or classification logits ``[B, T]``."""

        (
            x,
            edge_index,
            edge_attr,
            pair_index,
            spd,
            graph_batch,
            num_graphs,
        ) = self._batch_tensors(batch)
        nodes = self.node_encoder(x)
        edges = self.edge_encoder(edge_attr)
        globals_ = self.global_embedding.unsqueeze(0).expand(num_graphs, -1)

        for block in self.blocks:
            state = block(
                nodes,
                edges,
                globals_,
                edge_index,
                graph_batch,
                pair_index,
                spd,
            )
            nodes, edges, globals_ = state.nodes, state.edges, state.globals

        graph_nodes = scatter(
            nodes,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        return self.decoder(graph_nodes)

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Fetch and validate the 2D GPS++ graph and all-pair contract."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, pair_index, spd, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(pair_index, Tensor)
        assert isinstance(spd, Tensor)
        assert isinstance(graph_batch, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(
                f"batch.x must have shape [N, {self.atom_dim}] with N >= 1"
            )
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.edge_index must have shape [2, E] and dtype torch.long"
            )
        edge_count = edge_index.shape[1]
        if edge_attr.shape != (edge_count, self.bond_dim):
            raise ValueError(f"batch.edge_attr must have shape [E, {self.bond_dim}]")
        if edge_attr.dtype != torch.float32 or not torch.isfinite(edge_attr).all():
            raise ValueError("batch.edge_attr must contain finite torch.float32 values")
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if any(value.device != x.device for value in values):
            raise ValueError("all GPS++ batch tensors must share the node device")
        if graph_batch.numel() == 0 or graph_batch.min() < 0:
            raise ValueError("batch.batch must contain non-negative graph indices")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=True,
        )
        _validate_pyg_graph_order(graph_batch, num_graphs)
        _validate_reciprocal_edges(edge_index, edge_attr, num_nodes=x.shape[0])
        _validate_pair_contract(
            pair_index,
            spd,
            graph_batch,
            num_graphs=num_graphs,
            num_nodes=x.shape[0],
        )
        return x, edge_index, edge_attr, pair_index, spd, graph_batch, num_graphs


def _validate_reciprocal_edges(
    edge_index: Tensor, edge_attr: Tensor, *, num_nodes: int
) -> None:
    """Require one matching reverse for each supplied directed covalent edge."""

    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return
    source, target = edge_index
    encoded_pairs = source * num_nodes + target
    sorted_pairs, permutation = torch.sort(encoded_pairs)
    if bool((sorted_pairs[1:] == sorted_pairs[:-1]).any()):
        raise ValueError("batch.edge_index must not contain duplicate directed edges")

    reverse_pairs = target * num_nodes + source
    reverse_positions = torch.searchsorted(sorted_pairs, reverse_pairs)
    safe_positions = reverse_positions.clamp_max(edge_count - 1)
    found = (reverse_positions < edge_count) & (
        sorted_pairs[safe_positions] == reverse_pairs
    )
    if not bool(found.all()):
        raise ValueError(
            "batch.edge_index must contain one reciprocal directed edge for every edge"
        )
    reverse_edges = permutation[reverse_positions]
    if not torch.equal(edge_attr, edge_attr[reverse_edges]):
        raise ValueError("batch reciprocal directed edges must have matching edge_attr")


def _validate_pair_contract(
    pair_index: Tensor,
    spd: Tensor,
    graph_batch: Tensor,
    *,
    num_graphs: int,
    num_nodes: int,
) -> None:
    """Validate the all-pair shortest-path representation before attention."""

    if (
        pair_index.ndim != 2
        or pair_index.shape[0] != 2
        or pair_index.dtype != torch.long
    ):
        raise ValueError(
            "batch.gpspp_pair_index must have shape [2, P] and dtype torch.long"
        )
    pair_count = pair_index.shape[1]
    if spd.shape != (pair_count,) or spd.dtype != torch.long:
        raise ValueError("batch.gpspp_spd must have shape [P] and dtype torch.long")
    if pair_count == 0:
        raise ValueError("batch.gpspp_pair_index must include every self pair")
    if pair_index.min() < 0 or pair_index.max() >= num_nodes:
        raise ValueError("batch.gpspp_pair_index contains an invalid node index")
    if spd.min() < -1:
        raise ValueError("batch.gpspp_spd values must be -1 or non-negative")

    source, target = pair_index
    pair_graph = graph_batch[source]
    if not torch.equal(pair_graph, graph_batch[target]):
        raise ValueError("batch.gpspp_pair_index must not connect different graphs")
    counts = torch.bincount(graph_batch, minlength=num_graphs)
    expected_pair_count = int(counts.square().sum().item())
    if pair_count != expected_pair_count:
        raise ValueError(
            "batch.gpspp_pair_index must enumerate every ordered node pair exactly once"
        )

    self_pairs = source == target
    if not bool((spd[self_pairs] == 0).all()):
        raise ValueError("batch.gpspp_spd must be zero for every self pair")
    if bool((spd[~self_pairs] == 0).any()):
        raise ValueError("batch.gpspp_spd must not be zero for a non-self pair")

    starts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))
    for graph_index, (start, stop) in enumerate(
        zip(starts[:-1].tolist(), starts[1:].tolist(), strict=True)
    ):
        node_count = stop - start
        graph_pairs = pair_graph == graph_index
        if int(graph_pairs.sum().item()) != node_count * node_count:
            raise ValueError(
                "batch.gpspp_pair_index must enumerate every ordered node pair exactly once"
            )
        local_source = source[graph_pairs] - start
        local_target = target[graph_pairs] - start
        pair_ids = local_source * node_count + local_target
        if torch.unique(pair_ids).numel() != node_count * node_count:
            raise ValueError(
                "batch.gpspp_pair_index must not omit or duplicate an ordered node pair"
            )


def _validate_pyg_graph_order(graph_batch: Tensor, num_graphs: int) -> None:
    """Require the standard contiguous graph packing produced by PyG Batch."""

    counts = torch.bincount(graph_batch, minlength=num_graphs)
    expected = torch.repeat_interleave(
        torch.arange(num_graphs, dtype=torch.long, device=graph_batch.device), counts
    )
    if not torch.equal(graph_batch, expected):
        raise ValueError("batch.batch must group graph rows contiguously in PyG order")


def _drop_path_rate(block_index: int, depth: int, maximum: float) -> float:
    """Linearly increase attention/FFN stochastic depth across the stack."""

    if depth == 1:
        return 0.0
    return maximum * block_index / (depth - 1)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _dropout(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError(f"{name} must be a finite value in [0, 1)")


GPSPP = GPSPlusPlus


__all__ = ["GPSPP", "GPSPlusPlus"]
