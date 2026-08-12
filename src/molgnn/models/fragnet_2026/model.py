"""Architecture-only FragNet 2026 molecular property predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import FragNetLayerA

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "rrelu": nn.RReLU,
    "relu6": nn.ReLU6,
    "prelu": nn.PReLU,
    "leakyrelu": nn.LeakyReLU,
}


class FragNet(BaseMolecularModel):
    """Hierarchical bond-atom-fragment graph network.

    The model consumes a multi-graph view produced by ``add_fragnet_inputs``:
    the canonical atom graph, a BRICS fragment graph, a bond-line graph with
    cosine-angle edge features derived from supplied coordinates, and a
    bipartite fragment-connection graph.  Graph construction belongs to the
    transform; this module only performs the declared message passing.
    """

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "batch",
        "frag_index",
        "x_frags",
        "atom_to_fragment",
        "frag_batch",
        "edge_index_bonds_graph",
        "edge_attr_bonds",
        "frag_connection_features",
        "edge_index_fbonds",
        "edge_attr_fbonds",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        emb_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        drop_ratio: float = 0.15,
        num_frag_connection_features: int = 6,
        head_hidden_dims: tuple[int, ...] = (256, 256, 256, 256),
        head_act: str = "celu",
    ) -> None:
        super().__init__()

        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (emb_dim, "emb_dim"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (num_frag_connection_features, "num_frag_connection_features"),
        ):
            _positive_int(value, name)
        for h_dim in head_hidden_dims:
            _positive_int(h_dim, "head_hidden_dims")
        if (
            isinstance(drop_ratio, bool)
            or not isinstance(drop_ratio, (float, int))
            or not 0 <= drop_ratio < 1
        ):
            raise ValueError("drop_ratio must be in [0, 1)")
        if emb_dim % num_heads:
            raise ValueError("emb_dim must be divisible by num_heads")
        if head_act not in _ACTIVATIONS:
            raise ValueError(
                f"head_act must be one of {'|'.join(sorted(_ACTIVATIONS))}; got {head_act!r}"
            )

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_targets = num_targets
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.drop_ratio = float(drop_ratio)
        self.num_frag_connection_features = num_frag_connection_features
        self.head_hidden_dims = tuple(head_hidden_dims)
        self.head_act = head_act

        # First layer projects from raw feature dims to hidden dim.
        self.layers = nn.ModuleList()
        self.layers.append(
            FragNetLayerA(
                atom_in=atom_dim,
                atom_out=emb_dim,
                frag_in=atom_dim,
                frag_out=emb_dim,
                edge_in=bond_dim,
                edge_out=emb_dim,
                fedge_in=num_frag_connection_features,
                num_heads=num_heads,
                bond_edge_in=1,
                fbond_edge_in=num_frag_connection_features,
            )
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                FragNetLayerA(
                    atom_in=emb_dim,
                    atom_out=emb_dim,
                    frag_in=emb_dim,
                    frag_out=emb_dim,
                    edge_in=emb_dim,
                    edge_out=emb_dim,
                    fedge_in=emb_dim,
                    num_heads=num_heads,
                    bond_edge_in=1,
                    fbond_edge_in=num_frag_connection_features,
                )
            )

        self.dropout = nn.Dropout(self.drop_ratio)
        # Port of the upstream FTHead3: a 5-layer MLP [2*emb_dim, h1, h2, h3, h4,
        # num_targets] with per-layer Dropout + activation between layers,
        # matching the paper's optimized config (h1..h4=256, act="celu").
        head_dims = [2 * emb_dim, *self.head_hidden_dims, num_targets]
        self.head_predictor = nn.ModuleList(
            [nn.Linear(head_dims[i], head_dims[i + 1]) for i in range(len(head_dims) - 1)]
        )
        self.head_activation = _ACTIVATIONS[head_act]()

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary classification logits."""

        (
            x,
            edge_index,
            edge_attr,
            graph_batch,
            frag_index,
            x_frags,
            atom_to_fragment,
            frag_batch,
            edge_index_bonds_graph,
            edge_attr_bonds,
            frag_connection_features,
            edge_index_fbonds,
            edge_attr_fbonds,
            num_graphs,
        ) = self._batch_tensors(batch)

        atom_features = self.dropout(x)
        fragment_features = self.dropout(x_frags)
        bond_graph_features = edge_attr
        fbond_graph_features = frag_connection_features

        for layer in self.layers:
            atom_features, fragment_features, bond_graph_features, fbond_graph_features = layer(
                atom_features,
                edge_index,
                bond_graph_features,
                frag_index,
                fragment_features,
                atom_to_fragment,
                bond_graph_features,
                edge_index_bonds_graph,
                edge_attr_bonds,
                fbond_graph_features,
                edge_index_fbonds,
                edge_attr_fbonds,
            )
            atom_features = F.relu(self.dropout(atom_features))
            fragment_features = F.relu(self.dropout(fragment_features))
            bond_graph_features = F.relu(self.dropout(bond_graph_features))
            fbond_graph_features = F.relu(self.dropout(fbond_graph_features))

        atom_pool = scatter(
            atom_features,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        fragment_pool = scatter(
            fragment_features,
            frag_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        fused = torch.cat((atom_pool, fragment_pool), dim=-1)
        for layer in self.head_predictor[:-1]:
            fused = self.head_activation(self.dropout(layer(fused)))
        return self.head_predictor[-1](fused)

    def _batch_tensors(self, batch: Batch) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        int,
    ]:
        """Validate, fetch, and re-shape the FragNet batch tensors."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        (
            x,
            edge_index,
            edge_attr,
            graph_batch,
            frag_index,
            x_frags,
            atom_to_fragment,
            frag_batch,
            edge_index_bonds_graph,
            edge_attr_bonds,
            frag_connection_features,
            edge_index_fbonds,
            edge_attr_fbonds,
        ) = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(graph_batch, Tensor)
        assert isinstance(frag_index, Tensor)
        assert isinstance(x_frags, Tensor)
        assert isinstance(atom_to_fragment, Tensor)
        assert isinstance(frag_batch, Tensor)
        assert isinstance(edge_index_bonds_graph, Tensor)
        assert isinstance(edge_attr_bonds, Tensor)
        assert isinstance(frag_connection_features, Tensor)
        assert isinstance(edge_index_fbonds, Tensor)
        assert isinstance(edge_attr_fbonds, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(f"batch.x must have shape [N, {self.atom_dim}] with N >= 1")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")

        self._validate_edges(edge_index, edge_attr, "edge", x.shape[0], self.bond_dim)
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if frag_index.shape[0] != 2 or frag_index.dtype != torch.long:
            raise ValueError(
                "batch.frag_index must have shape [2, E_frag] and dtype torch.long"
            )
        if x_frags.shape[0] and frag_index.shape[1]:
            if frag_index.min().item() < 0 or frag_index.max().item() >= x_frags.shape[0]:
                raise ValueError(
                    "batch.frag_index indices must be in [0, num_fragments)"
                )
        if x_frags.shape[1] != self.atom_dim:
            raise ValueError(
                f"batch.x_frags must have shape [num_fragments, {self.atom_dim}]"
            )
        if x_frags.dtype != torch.float32 or not torch.isfinite(x_frags).all():
            raise ValueError("batch.x_frags must contain finite torch.float32 values")
        if atom_to_fragment.shape != (x.shape[0],) or atom_to_fragment.dtype != torch.long:
            raise ValueError(
                "batch.atom_to_fragment must have shape [N] and dtype torch.long"
            )
        if atom_to_fragment.min().item() < 0:
            raise ValueError("batch.atom_to_fragment indices must be non-negative")
        if x_frags.shape[0]:
            if atom_to_fragment.max().item() >= x_frags.shape[0]:
                raise ValueError(
                    "batch.atom_to_fragment indices must be < num_fragments"
                )
        if frag_batch.shape[0] != x_frags.shape[0] or frag_batch.dtype != torch.long:
            raise ValueError(
                "batch.frag_batch must have shape [num_fragments] and dtype torch.long"
            )

        # Bond-graph (line graph) edges: nodes are directed edges of the atom
        # graph; each node belongs to the same graph as its source atom.
        if edge_index_bonds_graph.shape[0] != 2 or edge_index_bonds_graph.dtype != torch.long:
            raise ValueError(
                "batch.edge_index_bonds_graph must have shape [2, E_b] and dtype torch.long"
            )
        if edge_attr_bonds.shape[0] != edge_index_bonds_graph.shape[1] or edge_attr_bonds.shape[1] != 1:
            raise ValueError(
                "batch.edge_attr_bonds must have shape [E_b, 1] matching edge_index_bonds_graph"
            )
        if edge_attr_bonds.dtype != torch.float32 or not torch.isfinite(edge_attr_bonds).all():
            raise ValueError("batch.edge_attr_bonds must contain finite torch.float32 values")

        # Fragment-bond bipartite edges: nodes are connections; each connection
        # belongs to its source connection's graph.
        if edge_index_fbonds.shape[0] != 2 or edge_index_fbonds.dtype != torch.long:
            raise ValueError(
                "batch.edge_index_fbonds must have shape [2, E_fb] and dtype torch.long"
            )
        if edge_attr_fbonds.shape[0] != edge_index_fbonds.shape[1]:
            raise ValueError("batch.edge_attr_fbonds row count must match edge_index_fbonds")
        if edge_attr_fbonds.shape[1] != self.num_frag_connection_features:
            raise ValueError(
                f"batch.edge_attr_fbonds must have width {self.num_frag_connection_features}"
            )
        if frag_connection_features.shape[1] != self.num_frag_connection_features:
            raise ValueError(
                f"batch.frag_connection_features must have width {self.num_frag_connection_features}"
            )
        if (
            frag_connection_features.dtype != torch.float32
            or not torch.isfinite(frag_connection_features).all()
        ):
            raise ValueError(
                "batch.frag_connection_features must contain finite torch.float32 values"
            )
        if edge_attr_fbonds.dtype != torch.float32 or not torch.isfinite(edge_attr_fbonds).all():
            raise ValueError("batch.edge_attr_fbonds must contain finite torch.float32 values")

        if any(value.device != x.device for value in values):
            raise ValueError("all FragNet batch tensors must be on the same device")

        # Atom graph: forbid_self_loops=False because the layer adds them.
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=False,
        )
        fragment_graph_count = validate_batched_molecular_graph(
            frag_index,
            frag_batch,
            num_nodes=x_frags.shape[0],
            device=x.device,
            edge_field="frag_index",
            forbid_self_loops=False,
        )
        if fragment_graph_count != num_graphs:
            raise ValueError("batch.frag_batch must cover the same graphs as batch.batch")
        if not torch.equal(frag_batch[atom_to_fragment], graph_batch):
            raise ValueError(
                "batch.atom_to_fragment must assign every atom to a fragment in its graph"
            )

        # Bond-graph: each node is a directed edge of the atom graph; assign
        # graph ids by the source-atom's graph id. Skip validation when the
        # bond-graph is empty (no inter-bond edges means num_nodes == 0).
        if edge_index.shape[1] and edge_index_bonds_graph.shape[1]:
            bond_graph_batch = graph_batch[edge_index[0]]
            validate_batched_molecular_graph(
                edge_index_bonds_graph,
                bond_graph_batch,
                num_nodes=edge_index.shape[1],
                device=x.device,
                edge_field="edge_index_bonds_graph",
                forbid_self_loops=False,
            )

        # Fragment-bond graph: source and target are both connection indices,
        # so the standard validate_batched_molecular_graph would work, but
        # the cross-graph property is structurally guaranteed by construction
        # (a connection only connects to connections sharing one of its
        # parent fragments, which all belong to the same molecule). The
        # shape / dtype / device checks above already gate the field.

        return (
            x,
            edge_index,
            edge_attr,
            graph_batch,
            frag_index,
            x_frags,
            atom_to_fragment,
            frag_batch,
            edge_index_bonds_graph,
            edge_attr_bonds,
            frag_connection_features,
            edge_index_fbonds,
            edge_attr_fbonds,
            num_graphs,
        )

    @staticmethod
    def _validate_edges(
        edge_index: Tensor,
        edge_attr: Tensor,
        field: str,
        num_nodes: int,
        bond_dim: int,
    ) -> None:
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                f"batch.{field}_index must have shape [2, E] and dtype torch.long"
            )
        if edge_attr.shape != (edge_index.shape[1], bond_dim):
            raise ValueError(
                f"batch.{field}_attr must have shape [E, {bond_dim}]"
            )
        if edge_attr.dtype != torch.float32 or not torch.isfinite(edge_attr).all():
            raise ValueError(
                f"batch.{field}_attr must contain finite torch.float32 values"
            )
        if edge_index.shape[1] and (
            edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes
        ):
            raise ValueError(f"batch.{field}_index contains an invalid node index")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["FragNet"]
