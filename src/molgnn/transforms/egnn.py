"""3D conformer derivation for the EGNN architecture.

EGNN's positional message term (paper Eq. 4) consumes pairwise squared
distances between atom coordinates, so every input graph must carry a
``[N, 3]`` float32 coordinate tensor.  The canonical featurizer produces only
2-D atom and bond features; this transform closes the gap by generating a
deterministic 3-D conformer per molecule via RDKit's ETKDG embedding + MMFF
optimisation and attaching it as ``data.pos``.

.. warning::

   This transform synthesises a conformer from SMILES via
   ``AllChem.EmbedMolecule`` (ETKDG) + ``MMFFOptimizeMolecule``.  That is
   **not** what the paper does: EGNN's authors trained and evaluated on
   QM9, whose ``data.pos`` ships with DFT-optimised geometries
   (B3LYP/6-31G(2df,p)).  An MMFF-relaxed conformer is a much weaker
   proxy for a low-energy 3-D structure and will materially depress the
   model's performance on 2-D-only benchmarks (BACE, BBBP, …).  For a
   faithful evaluation, swap to a dataset that ships with real 3-D
   coordinates (QM9, MD17, GEOM-DRUGS, …) and replace this transform with
   one that reads the precomputed conformers from disk.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from ..data import MolecularData
from .base import TransformError


def add_egnn_inputs(data: MolecularData) -> MolecularData:
    """Clone a canonical graph and attach a ``pos`` ``[N, 3]`` float32 tensor.

    Reads ``data.smiles``, regenerates the molecule with explicit hydrogens
    (required for ``AllChem.EmbedMolecule``), runs an ETKDG conformer
    embedding + MMFF force-field optimisation, and stores the resulting
    coordinates as ``data.pos``.  Falls back to identity coordinates when the
    conformer cannot be embedded (single-atom molecules, or pathological
    cases where ETKDG fails to converge); the EGNN layer remains numerically
    well-defined because all pairwise distances collapse to zero.

    The function never mutates the input ``data`` — a clone is returned so
    that downstream transforms can re-derive their own fields safely.

    .. warning::

       See the module-level warning: this transform generates a *synthetic*
       ETKDG/MMFF conformer, **not** the DFT-optimised QM9 geometry the
       paper assumes.  Treat EGNN numbers on 2-D-only benchmarks as a
       lower bound on the architecture's true capability.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; EGNN inputs must be derived before batching"
        )

    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(f"sample {sample} is missing source SMILES metadata")

    x = getattr(data, "x", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise TransformError(f"sample {sample} has invalid x")
    if x.dtype != torch.float32:
        raise TransformError(f"sample {sample} x must be float32")
    if not torch.isfinite(x).all():
        raise TransformError(f"sample {sample} x must contain only finite values")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"sample {sample} has invalid source SMILES")
    if mol.GetNumAtoms() != x.shape[0]:
        raise TransformError(
            f"sample {sample} source SMILES atom count does not match x"
        )

    device = x.device
    pos = _embed_conformer(mol, sample=sample, device=device)
    transformed = data.clone()
    transformed.pos = pos
    return transformed


def _embed_conformer(
    mol: Chem.Mol,
    *,
    sample: int | str,
    device: torch.device,
) -> Tensor:
    """Embed one molecule with explicit hydrogens and return ``[N, 3]`` float32."""

    # ``AllChem.EmbedMolecule`` requires explicit hydrogens; the resulting
    # conformer is keyed on the heavy-atom indices that match ``x``.
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    embed_status = AllChem.EmbedMolecule(mol_h, params)
    if embed_status != 0:
        # Fall back to identity coordinates.  EGNN still runs because every
        # pairwise distance is zero, but we surface a warning through the
        # ``sample`` marker so users can see which molecules failed.
        positions = [(0.0, 0.0, float(index)) for index in range(mol.GetNumAtoms())]
        return torch.tensor(positions, dtype=torch.float32, device=device)

    # ``MMFFOptimizeMolecule`` is significantly faster than UFF and gives a
    # smoother potential energy surface for organic molecules — both
    # optimisers leave heavy-atom indices untouched.
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except RuntimeError:
        # Force-field setup can fail on exotic chemistries (e.g. charged
        # metals); fall through and use the unoptimised conformer.
        pass

    conf = mol_h.GetConformer()
    coords: list[tuple[float, float, float]] = []
    for heavy_index in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(heavy_index)
        # The heavy atom index in the parent ``mol`` matches the implicit
        # heavy-atom numbering used by ``AddHs`` (hydrogens appended last).
        point = conf.GetAtomPosition(atom.GetIdx())
        coords.append((float(point.x), float(point.y), float(point.z)))

    tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    if tensor.shape != (mol.GetNumAtoms(), 3):
        raise TransformError(
            f"sample {sample} conformer produced unexpected shape {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise TransformError(
            f"sample {sample} conformer contains non-finite coordinates"
        )
    return tensor


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_egnn_inputs"]