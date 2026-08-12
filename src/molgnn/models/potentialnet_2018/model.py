"""PotentialNet's full staged protein-ligand graph architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import LigandReadout, TypedRecurrentStage


class PotentialNet(BaseMolecularModel):
    """Staged, typed graph propagation for molecular graphs and complexes.

    Stage 1 sees only typed covalent edges.  Stage 2 starts from Stage 1's
    atom embeddings and sees the full typed graph of covalent and spatial
    edges when spatial input is available.  For paper-style 2D use, Stage 2
    is skipped and the Stage 1 atom embeddings flow directly to the
    ligand-only readout.

    ``spatial_mode='auto'`` preserves direct 3D callers: paired Stage 2
    tensors activate the spatial branch even without an explicit marker.  A
    transform or dataset may instead supply ``potentialnet_use_spatial`` as
    one homogeneous boolean per graph to make the selected branch explicit.
    """

    required_batch_fields = (
        "x",
        "potentialnet_bond_edge_index",
        "potentialnet_bond_edge_type",
        "ligand_mask",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_hidden_dim: int = 256,
        spatial_hidden_dim: int = 256,
        gather_dim: int = 256,
        num_bond_edge_types: int = 5,
        num_stage2_edge_types: int = 9,
        num_bond_steps: int = 2,
        num_spatial_steps: int = 2,
        message_hidden_dim: int | None = None,
        readout_hidden_dims: Sequence[int] = (128, 32),
        num_targets: int = 1,
        dropout: float = 0.0,
        spatial_mode: str = "auto",
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_hidden_dim, "bond_hidden_dim"),
            (spatial_hidden_dim, "spatial_hidden_dim"),
            (gather_dim, "gather_dim"),
            (num_bond_edge_types, "num_bond_edge_types"),
            (num_stage2_edge_types, "num_stage2_edge_types"),
            (num_bond_steps, "num_bond_steps"),
            (num_spatial_steps, "num_spatial_steps"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        if message_hidden_dim is not None:
            _positive_int(message_hidden_dim, "message_hidden_dim")
        _dropout(dropout)
        if spatial_mode not in {"auto", "required", "disabled"}:
            raise ValueError("spatial_mode must be one of: auto, required, disabled")
        if bond_hidden_dim < atom_dim:
            raise ValueError(
                "bond_hidden_dim must be at least atom_dim for zero-padded Stage 1 h0"
            )
        if spatial_hidden_dim < gather_dim:
            raise ValueError(
                "spatial_hidden_dim must be at least gather_dim for zero-padded Stage 2 h0"
            )
        if num_stage2_edge_types < num_bond_edge_types:
            raise ValueError(
                "num_stage2_edge_types must include all num_bond_edge_types"
            )

        bond_message_hidden_dim = message_hidden_dim or bond_hidden_dim
        spatial_message_hidden_dim = message_hidden_dim or spatial_hidden_dim
        self.atom_dim = atom_dim
        self.bond_hidden_dim = bond_hidden_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.gather_dim = gather_dim
        self.num_bond_edge_types = num_bond_edge_types
        self.num_stage2_edge_types = num_stage2_edge_types
        self.num_bond_steps = num_bond_steps
        self.num_spatial_steps = num_spatial_steps
        self.message_hidden_dim = message_hidden_dim
        self.num_targets = num_targets
        self.spatial_mode = spatial_mode

        self.stage1 = TypedRecurrentStage(
            input_dim=atom_dim,
            state_dim=bond_hidden_dim,
            gather_dim=gather_dim,
            num_edge_types=num_bond_edge_types,
            num_steps=num_bond_steps,
            message_hidden_dim=bond_message_hidden_dim,
            dropout=dropout,
        )
        self.stage2 = TypedRecurrentStage(
            input_dim=gather_dim,
            state_dim=spatial_hidden_dim,
            gather_dim=gather_dim,
            num_edge_types=num_stage2_edge_types,
            num_steps=num_spatial_steps,
            message_hidden_dim=spatial_message_hidden_dim,
            dropout=dropout,
        )
        self.readout = LigandReadout(
            gather_dim, readout_hidden_dims, num_targets, dropout
        )

    def reset_parameters(self) -> None:
        """Reset both independent recurrent stages and the final predictor."""

        self.stage1.reset_parameters()
        self.stage2.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or classification logits."""

        (
            x,
            bond_edge_index,
            bond_edge_type,
            stage2_edge_index,
            stage2_edge_type,
            ligand_mask,
            graph_batch,
            num_graphs,
            use_spatial,
        ) = self._batch_tensors(batch)
        bond_embeddings = self.stage1(x, bond_edge_index, bond_edge_type)
        node_embeddings = bond_embeddings
        if use_spatial:
            assert isinstance(stage2_edge_index, Tensor)
            assert isinstance(stage2_edge_type, Tensor)
            node_embeddings = self.stage2(
                bond_embeddings, stage2_edge_index, stage2_edge_type
            )
        return self.readout(node_embeddings, ligand_mask, graph_batch, num_graphs)

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor,
        Tensor,
        int,
        bool,
    ]:
        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        (
            x,
            bond_edge_index,
            bond_edge_type,
            ligand_mask,
            graph_batch,
        ) = values
        assert isinstance(x, Tensor)
        assert isinstance(bond_edge_index, Tensor)
        assert isinstance(bond_edge_type, Tensor)
        assert isinstance(ligand_mask, Tensor)
        assert isinstance(graph_batch, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(
                f"batch.x must have shape [N, {self.atom_dim}] with N >= 1"
            )
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")
        if ligand_mask.shape != (x.shape[0],) or ligand_mask.dtype != torch.bool:
            raise ValueError(
                "batch.ligand_mask must have shape [N] and dtype torch.bool"
            )
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if any(value.device != x.device for value in values):
            raise ValueError(
                "all PotentialNet batch tensors must share the node device"
            )

        self._validate_edge_tensors(
            bond_edge_index,
            bond_edge_type,
            edge_index_field="potentialnet_bond_edge_index",
            edge_type_field="potentialnet_bond_edge_type",
            num_edge_types=self.num_bond_edge_types,
            num_nodes=x.shape[0],
        )
        if graph_batch.numel() == 0 or graph_batch.min() < 0:
            raise ValueError("batch.batch must contain non-negative graph indices")
        num_graphs = validate_batched_molecular_graph(
            bond_edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="potentialnet_bond_edge_index",
            forbid_self_loops=True,
        )
        ligand_counts = scatter(
            ligand_mask.to(dtype=torch.long),
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        if torch.any(ligand_counts == 0):
            raise ValueError(
                "each PotentialNet graph must contain at least one ligand atom"
            )

        stage2_edge_index, stage2_edge_type = self._optional_stage2_tensors(batch)
        if stage2_edge_index is not None and stage2_edge_type is not None:
            if (
                stage2_edge_index.device != x.device
                or stage2_edge_type.device != x.device
            ):
                raise ValueError(
                    "PotentialNet Stage 2 tensors must share the node device"
                )
            self._validate_edge_tensors(
                stage2_edge_index,
                stage2_edge_type,
                edge_index_field="potentialnet_stage2_edge_index",
                edge_type_field="potentialnet_stage2_edge_type",
                num_edge_types=self.num_stage2_edge_types,
                num_nodes=x.shape[0],
            )
        use_spatial = self._resolve_spatial_branch(
            batch,
            has_stage2_fields=stage2_edge_index is not None,
            stage2_edge_count=(
                None if stage2_edge_index is None else stage2_edge_index.shape[1]
            ),
            num_graphs=num_graphs,
            device=x.device,
        )
        if use_spatial:
            assert isinstance(stage2_edge_index, Tensor)
            assert isinstance(stage2_edge_type, Tensor)
            stage2_num_graphs = validate_batched_molecular_graph(
                stage2_edge_index,
                graph_batch,
                num_nodes=x.shape[0],
                device=x.device,
                edge_field="potentialnet_stage2_edge_index",
                forbid_self_loops=True,
            )
            if stage2_num_graphs != num_graphs:
                raise RuntimeError(
                    "PotentialNet edge partitions disagree on graph count"
                )
        return (
            x,
            bond_edge_index,
            bond_edge_type,
            stage2_edge_index,
            stage2_edge_type,
            ligand_mask,
            graph_batch,
            num_graphs,
            use_spatial,
        )

    @staticmethod
    def _optional_stage2_tensors(batch: Batch) -> tuple[Tensor | None, Tensor | None]:
        """Return the paired optional Stage 2 tensors or reject partial input."""

        edge_index = getattr(batch, "potentialnet_stage2_edge_index", None)
        edge_type = getattr(batch, "potentialnet_stage2_edge_type", None)
        if (edge_index is None) != (edge_type is None):
            raise ValueError(
                "batch.potentialnet_stage2_edge_index and "
                "batch.potentialnet_stage2_edge_type must be provided together"
            )
        if edge_index is None:
            return None, None
        if not isinstance(edge_index, Tensor) or not isinstance(edge_type, Tensor):
            raise ValueError(
                "batch.potentialnet_stage2_edge_index and "
                "batch.potentialnet_stage2_edge_type must be torch.Tensor values"
            )
        return edge_index, edge_type

    def _resolve_spatial_branch(
        self,
        batch: Batch,
        *,
        has_stage2_fields: bool,
        stage2_edge_count: int | None,
        num_graphs: int,
        device: torch.device,
    ) -> bool:
        """Resolve the configured branch without allowing mixed-complex batches."""

        marker = getattr(batch, "potentialnet_use_spatial", None)
        marked_spatial: bool | None = None
        if marker is not None:
            if not isinstance(marker, Tensor):
                raise ValueError(
                    "batch.potentialnet_use_spatial must be a torch.Tensor"
                )
            if marker.dtype != torch.bool or marker.device != device:
                raise ValueError(
                    "batch.potentialnet_use_spatial must be a bool tensor on the node device"
                )
            if marker.ndim == 0 and num_graphs == 1:
                marker = marker.reshape(1)
            if marker.shape != (num_graphs,):
                raise ValueError(
                    "batch.potentialnet_use_spatial must have shape [num_graphs]"
                )
            if not torch.equal(marker, marker[:1].expand_as(marker)):
                raise ValueError(
                    "batch.potentialnet_use_spatial must be homogeneous across a batch"
                )
            marked_spatial = bool(marker[0].item())
            if marked_spatial and not has_stage2_fields:
                raise ValueError(
                    "batch.potentialnet_use_spatial=True requires paired Stage 2 tensors"
                )
            if not marked_spatial and stage2_edge_count not in (None, 0):
                raise ValueError(
                    "batch.potentialnet_use_spatial=False only permits empty "
                    "Stage 2 tensors"
                )

        use_spatial = has_stage2_fields if marked_spatial is None else marked_spatial
        if self.spatial_mode == "required" and not use_spatial:
            raise ValueError(
                "spatial_mode='required' requires paired PotentialNet Stage 2 tensors"
            )
        if self.spatial_mode == "disabled":
            if use_spatial:
                raise ValueError(
                    "spatial_mode='disabled' does not accept PotentialNet Stage 2 tensors"
                )
            return False
        return use_spatial

    @staticmethod
    def _validate_edge_tensors(
        edge_index: Tensor,
        edge_type: Tensor,
        *,
        edge_index_field: str,
        edge_type_field: str,
        num_edge_types: int,
        num_nodes: int,
    ) -> int:
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                f"batch.{edge_index_field} must have shape [2, E] and dtype torch.long"
            )
        edge_count = edge_index.shape[1]
        if edge_type.shape != (edge_count,) or edge_type.dtype != torch.long:
            raise ValueError(
                f"batch.{edge_type_field} must have shape [E] and dtype torch.long"
            )
        if edge_count and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
            raise ValueError(f"batch.{edge_index_field} contains an invalid node index")
        if edge_count and (edge_type.min() < 0 or edge_type.max() >= num_edge_types):
            raise ValueError(f"batch.{edge_type_field} contains an invalid edge type")
        return edge_count


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _dropout(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not 0 <= value < 1
    ):
        raise ValueError("dropout must be in [0, 1)")


__all__ = ["PotentialNet"]
