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

import math

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, rdMolTransforms
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

    Reads ``data.smiles``, ``data.x``, ``data.edge_index`` and
    ``data.edge_attr`` and sets the following fields on the returned clone:

    * ``frag_index`` / ``atom_to_fragment`` / ``x_frags`` / ``frag_batch``:
      fragment graph derived from BRICS connected components,
    * ``edge_index_bonds_graph`` / ``edge_attr_bonds``: bond (line) graph with
      ``cos(bond angle)`` edge features from a deterministic 3D conformer,
    * ``frag_connection_features`` / ``edge_index_fbonds`` /
      ``edge_attr_fbonds``: bipartite fragment-connection graph.

    All produced tensors live on the same device as ``data.x``.
    """

    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(f"sample {_sample_id(data)} is missing source SMILES metadata")

    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if not isinstance(x, Tensor) or x.ndim != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid x")
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_index")
    if edge_index.dtype != torch.long:
        raise TransformError(f"sample {_sample_id(data)} edge_index must have dtype torch.long")
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_attr")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise TransformError(f"sample {_sample_id(data)} has mismatched edge feature count")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"sample {_sample_id(data)} has invalid source SMILES")
    if mol.GetNumAtoms() != x.shape[0]:
        raise TransformError(f"sample {_sample_id(data)} source SMILES atom count does not match x")

    device = x.device
    num_atoms = int(x.shape[0])
    edge_src = edge_index[0]
    edge_dst = edge_index[1]

    # ---- 3D conformer for bond-angle features (None if embedding fails). ----
    # Reuse the parsed mol so we don't run RDKit parsing twice for each sample.
    conf = _embed_conformer(mol)

    # ---- Fragment graph from BRICS connected components. ---------------------
    cut_bonds = frozenset(frozenset(pair) for pair, _labels in BRICS.FindBRICSBonds(mol))
    atom_to_fragment = _connected_components(mol, set(cut_bonds), device=device)
    num_fragments = int(atom_to_fragment.max().item()) + 1 if num_atoms else 0

    if num_atoms:
        x_frags = scatter(x, atom_to_fragment, dim=0, reduce="sum", dim_size=num_fragments)
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
        edge_index_bonds_graph = torch.tensor([lb_src, lb_dst], dtype=torch.long, device=device)
    else:
        edge_index_bonds_graph = torch.empty((2, 0), dtype=torch.long, device=device)

    if conf is None or num_line_edges == 0:
        edge_attr_bonds = torch.zeros((num_line_edges, 1), dtype=torch.float32, device=device)
    else:
        angle_features: list[float] = []
        for e1, e2, i in zip(lb_src, lb_dst, lb_atom):
            j = int(edge_dst[e1])
            k = int(edge_src[e2])
            if j == k:
                # The two directed bonds are the reverse of the same bond: the
                # angle is undefined for a one-bond fragment, so use 1.0.
                angle_features.append(1.0)
            else:
                angle = rdMolTransforms.GetAngleRad(conf, j, i, k)
                angle_features.append(math.cos(angle))
        edge_attr_bonds = torch.tensor(
            angle_features, dtype=torch.float32, device=device
        ).reshape(-1, 1)

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
        edge_index_fbonds = torch.tensor([fb_src, fb_dst], dtype=torch.long, device=device)
        edge_attr_fbonds = torch.stack(fb_attr, dim=0) if fb_attr else torch.empty(
            (0, 6), dtype=torch.float32, device=device
        )
    else:
        frag_connection_features = torch.empty((0, 6), dtype=torch.float32, device=device)
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


def _embed_conformer(mol: Chem.Mol) -> Chem.Conformer | None:
    """Embed a deterministic 3D conformer for bond-angle features.

    Uses ETKDG with a fixed seed (falling back to a second fixed seed). Returns
    ``None`` when no conformer could be embedded; callers then fall back to
    zero bond-angle features.
    """

    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol_h, params)
    if status == -1:
        params = AllChem.ETKDGv3()
        params.randomSeed = 1
        status = AllChem.EmbedMolecule(mol_h, params)
    if status == -1:
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:  # noqa: BLE001 - 3D geometry is best-effort here
        pass
    return mol_h.GetConformer()


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
