"""FragNet multi-graph view derived from a canonical :class:`MolecularData`.

FragNet consumes four coupled graphs per sample:

* the atom graph (the canonical graph, unchanged),
* a fragment graph whose nodes aggregate atom features per BRICS fragment,
* a bond graph whose nodes are the directed edges of the atom graph and whose
  edges connect bonds that share an atom (a line graph),
* a fragment-bond graph whose nodes are inter-fragment connections and whose
  edges connect pairs of connections sharing a fragment (mirroring upstream
  ``get_fragbond``).

This module only *derives* the view; the model and its registry wiring are
added separately.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch import Tensor
from torch_geometric.utils import scatter

from ..data import MolecularData
from .base import TransformError

# Canonical bond-type ordering used by the 4-dim bond-type one-hot block of
# the fragment-connection features. Matches the featurizer's BOND_TYPE_VOCAB.
_BOND_TYPE_ORDER: tuple[Chem.BondType, ...] = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)


def add_fragnet_inputs(data: MolecularData) -> MolecularData:
    """Clone a canonical graph and attach its FragNet multi-graph view.

    Reads ``data.smiles``, ``data.x``, ``data.edge_index``, ``data.edge_attr``
    and supplied ``data.pos`` coordinates, then sets the following fields on
    the returned clone:

    * ``frag_index`` / ``atom_to_fragment`` / ``x_frags`` / ``frag_batch``:
      fragment graph derived from BRICS connected components,
    * ``edge_index_bonds_graph`` / ``edge_attr_bonds``: bond (line) graph with
      ``cos(bond angle)`` edge features from the supplied coordinates,
    * ``frag_connection_features`` / ``edge_index_fbonds`` /
      ``edge_attr_fbonds``: bipartite fragment-connection graph.

    All produced tensors live on the same device as ``data.x``.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; FragNet inputs must be derived before batching"
        )
    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(f"sample {sample} is missing source SMILES metadata")

    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    pos = getattr(data, "pos", None)
    if not isinstance(x, Tensor) or x.ndim != 2:
        raise TransformError(f"sample {sample} has invalid x")
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise TransformError(f"sample {sample} has invalid edge_index")
    if edge_index.dtype != torch.long:
        raise TransformError(f"sample {sample} edge_index must have dtype torch.long")
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2:
        raise TransformError(f"sample {sample} has invalid edge_attr")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise TransformError(f"sample {sample} has mismatched edge feature count")
    if (
        not isinstance(pos, Tensor)
        or pos.shape != (x.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != x.device
        or not torch.isfinite(pos).all()
    ):
        raise TransformError(
            f"sample {sample} requires finite float32 pos with shape [N, 3]"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"sample {sample} has invalid source SMILES")
    if mol.GetNumAtoms() != x.shape[0]:
        raise TransformError(
            f"sample {sample} source SMILES atom count does not match x"
        )

    device = x.device
    num_atoms = int(x.shape[0])
    edge_src = edge_index[0]
    edge_dst = edge_index[1]

    # ---- Fragment graph from BRICS connected components. ---------------------
    cut_bonds = frozenset(
        frozenset(pair) for pair, _labels in BRICS.FindBRICSBonds(mol)
    )
    atom_to_fragment = _connected_components(mol, set(cut_bonds), device=device)
    num_fragments = int(atom_to_fragment.max().item()) + 1 if num_atoms else 0

    if num_atoms:
        x_frags = scatter(
            x, atom_to_fragment, dim=0, reduce="sum", dim_size=num_fragments
        )
    else:
        x_frags = torch.empty((0, x.shape[1]), dtype=torch.float32, device=device)
    frag_batch = torch.zeros(num_fragments, dtype=torch.long, device=device)

    # ---- Fragment-graph edges and fragment-connection graph. -----------------
    # Iterate over canonical (min, max) bond pairs once. Each cross-fragment
    # pair yields:
    #   * one fragment-to-fragment bidirectional edge (BeginFrag ↔ EndFrag)
    #     for the fragment-graph GAT,
    #   * one fragment-connection node with a 6-dim feature,
    #   * two later-emitted edges connecting pairs of connections that share
    #     a fragment, for the fragment-bond graph.
    seen_pairs: set[tuple[int, int]] = set()
    fragment_pairs: list[tuple[int, int]] = []
    connections: list[tuple[int, int, Tensor]] = []
    for column in range(int(edge_index.shape[1])):
        a = int(edge_src[column])
        b = int(edge_dst[column])
        key = (min(a, b), max(a, b))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        fragment_a = int(atom_to_fragment[a])
        fragment_b = int(atom_to_fragment[b])
        if fragment_a == fragment_b:
            continue
        fragment_pairs.append((fragment_a, fragment_b))
        bond = mol.GetBondBetweenAtoms(a, b)
        bond_type = bond.GetBondType() if bond is not None else Chem.BondType.SINGLE
        feature = _connection_feature(bond_type, device=device)
        connections.append((fragment_a, fragment_b, feature))

    if fragment_pairs:
        f_src: list[int] = []
        f_dst: list[int] = []
        for fragment_a, fragment_b in fragment_pairs:
            f_src.extend((fragment_a, fragment_b))
            f_dst.extend((fragment_b, fragment_a))
        frag_index = torch.tensor([f_src, f_dst], dtype=torch.long, device=device)
    else:
        # No cross-fragment bonds: keep empty fragment-graph edges and empty
        # fragment-bond-graph. The layer's fragment-graph GAT skips when empty.
        frag_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # ---- Bond (line) graph. --------------------------------------------------
    # One node per directed edge of the atom graph. Two bond-nodes are linked
    # when they share an atom; for atom i every (source-edge, target-edge) pair
    # incident on i forms one line-graph edge.
    lb_src: list[int] = []
    lb_dst: list[int] = []
    lb_atom: list[int] = []
    for i in range(num_atoms):
        src_here = (edge_src == i).nonzero(as_tuple=False).flatten().tolist()
        dst_here = (edge_dst == i).nonzero(as_tuple=False).flatten().tolist()
        for e1 in src_here:
            for e2 in dst_here:
                if e1 != e2:  # skip self-loops (should not occur anyway)
                    lb_src.append(e1)
                    lb_dst.append(e2)
                    lb_atom.append(i)

    num_line_edges = len(lb_src)
    if num_line_edges:
        edge_index_bonds_graph = torch.tensor(
            [lb_src, lb_dst], dtype=torch.long, device=device
        )
    else:
        edge_index_bonds_graph = torch.empty((2, 0), dtype=torch.long, device=device)

    edge_attr_bonds = _bond_angle_features(
        pos,
        lb_src,
        lb_dst,
        lb_atom,
        edge_src,
        edge_dst,
        sample=sample,
    )

    # ---- Fragment-connection graph (connection ↔ connection edges). -----------
    # The upstream builds a graph whose nodes are connections and whose edges
    # connect pairs of connections that share a fragment. For each sharing
    # pair (ie, ke) we emit two directed edges (ie→ke, ke→ie) with the
    # element-wise sum of the two connections' features as the edge attribute.
    num_connections = len(connections)
    if num_connections:
        frag_connection_features = torch.stack([c[2] for c in connections], dim=0)
        # Map each fragment to the list of connections containing it.
        fragment_to_connections: dict[int, list[int]] = {}
        for connection_id, (fragment_a, fragment_b, _feature) in enumerate(connections):
            fragment_to_connections.setdefault(fragment_a, []).append(connection_id)
            fragment_to_connections.setdefault(fragment_b, []).append(connection_id)
        fb_src: list[int] = []
        fb_dst: list[int] = []
        fb_attr: list[Tensor] = []
        for _fragment, conn_ids in fragment_to_connections.items():
            for i_idx in range(len(conn_ids)):
                for j_idx in range(len(conn_ids)):
                    if i_idx == j_idx:
                        continue
                    ie, je = conn_ids[i_idx], conn_ids[j_idx]
                    fb_src.append(ie)
                    fb_dst.append(je)
                    fb_attr.append(
                        frag_connection_features[ie] + frag_connection_features[je]
                    )
        edge_index_fbonds = torch.tensor(
            [fb_src, fb_dst], dtype=torch.long, device=device
        )
        edge_attr_fbonds = (
            torch.stack(fb_attr, dim=0)
            if fb_attr
            else torch.empty((0, 6), dtype=torch.float32, device=device)
        )
    else:
        frag_connection_features = torch.empty(
            (0, 6), dtype=torch.float32, device=device
        )
        edge_index_fbonds = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr_fbonds = torch.empty((0, 6), dtype=torch.float32, device=device)

    transformed = data.clone()
    transformed.frag_index = frag_index
    transformed.atom_to_fragment = atom_to_fragment
    transformed.x_frags = x_frags
    transformed.frag_batch = frag_batch
    transformed.edge_index_bonds_graph = edge_index_bonds_graph
    transformed.edge_attr_bonds = edge_attr_bonds
    transformed.frag_connection_features = frag_connection_features
    transformed.edge_index_fbonds = edge_index_fbonds
    transformed.edge_attr_fbonds = edge_attr_fbonds
    return transformed


def _connection_feature(bond_type: Chem.BondType, *, device: torch.device) -> Tensor:
    """Build the 6-dim FragNet connection feature for one inter-fragment bond.

    Layout: 4-dim bond-type one-hot, ``is_self_cn``, ``is_iso_cn3``. Both
    connection flags are always zero in the current view.
    """

    one_hot = [0.0] * 6
    if bond_type in _BOND_TYPE_ORDER:
        one_hot[_BOND_TYPE_ORDER.index(bond_type)] = 1.0
    return torch.tensor(one_hot, dtype=torch.float32, device=device)


def _bond_angle_features(
    pos: Tensor,
    line_sources: list[int],
    line_targets: list[int],
    centers: list[int],
    edge_source: Tensor,
    edge_target: Tensor,
    *,
    sample: int | str,
) -> Tensor:
    """Return cosine-angle attributes for the directed bond-line graph."""

    line_edge_count = len(line_sources)
    if not line_edge_count:
        return torch.empty((0, 1), dtype=torch.float32, device=pos.device)
    first_atoms = torch.tensor(
        [int(edge_target[index]) for index in line_sources],
        dtype=torch.long,
        device=pos.device,
    )
    second_atoms = torch.tensor(
        [int(edge_source[index]) for index in line_targets],
        dtype=torch.long,
        device=pos.device,
    )
    center_atoms = torch.tensor(centers, dtype=torch.long, device=pos.device)
    repeated_bond = first_atoms == second_atoms
    features = torch.ones(line_edge_count, dtype=torch.float32, device=pos.device)
    valid = ~repeated_bond
    if not bool(valid.any()):
        return features.unsqueeze(-1)

    first_vectors = pos[first_atoms[valid]] - pos[center_atoms[valid]]
    second_vectors = pos[second_atoms[valid]] - pos[center_atoms[valid]]
    denominator = torch.linalg.vector_norm(
        first_vectors, dim=-1
    ) * torch.linalg.vector_norm(second_vectors, dim=-1)
    if bool((denominator == 0).any()):
        raise TransformError(
            f"sample {sample} has coincident atoms in a FragNet bond angle"
        )
    cosine = (first_vectors * second_vectors).sum(dim=-1) / denominator
    features[valid] = cosine.clamp(-1, 1)
    return features.unsqueeze(-1)


def _connected_components(
    mol: Chem.Mol,
    cut_bonds: set[frozenset[int]],
    *,
    device: torch.device,
) -> Tensor:
    """Assign every atom to a fragment id over the graph minus the cut bonds."""

    adjacency: list[list[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        if frozenset((source, target)) in cut_bonds:
            continue
        adjacency[source].append(target)
        adjacency[target].append(source)

    assignments = [-1] * mol.GetNumAtoms()
    fragment = 0
    for root in range(mol.GetNumAtoms()):
        if assignments[root] >= 0:
            continue
        assignments[root] = fragment
        stack = [root]
        while stack:
            atom = stack.pop()
            for neighbor in adjacency[atom]:
                if assignments[neighbor] < 0:
                    assignments[neighbor] = fragment
                    stack.append(neighbor)
        fragment += 1
    return torch.tensor(assignments, dtype=torch.long, device=device)


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_fragnet_inputs"]
