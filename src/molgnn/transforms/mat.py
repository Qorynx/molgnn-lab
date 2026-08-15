"""Dummy node + 3D conformer derivation for the MAT architecture.

MAT (Maziarka et al., 2020) augments the Transformer encoder with a
disconnected "dummy" node per molecule (Li et al., 2017; Clark et al.,
2019) and with inter-atomic distance matrices (paper Eq. 2). The
canonical featurizer emits 2-D atom and bond features only; this transform
closes the gap by

  1. embedding an RDKit ETKDG conformer (with UFF optimisation) so the
     model can later build a pairwise distance matrix;
  2. prepending a dummy atom at index 0 with a zero feature vector
     marked by ``x[0, 0] = 1`` (matches the upstream ``featurize_mol``
     convention of setting the dummy's first feature to ``1.0``);
  3. attaching the 3-D coordinates as ``data.pos`` — the dummy gets
     ``pos = (1e6, 1e6, 1e6)`` so the model sees "very far" distances to
     every real atom, matching the upstream's ``1e6``-padding convention.

No edge is added from the dummy to any real atom — the dummy stays
disconnected, which is what the paper intends.

.. warning::

   The upstream trains and evaluates on 7 MoleculeNet benchmarks whose
   3-D conformers were pre-computed with RDKit ``UFFOptimizeMolecule``.
   The lab's ``add_mat_inputs`` transform runs that same routine on the
   fly per molecule.  Performance numbers from BACE should therefore be
   read as a lower bound on the architecture's true capability, not as
   an indictment of the model code.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from ..data import MolecularData
from .base import TransformError

# Mirror the upstream ``featurize_mol`` padding for the dummy node's
# distance-to-self / distance-to-others — large enough that softmax(-D)
# puts effectively zero weight on the dummy in the attention kernel.
_DUMMY_DISTANCE = 1.0e6


def add_mat_inputs(data: MolecularData) -> MolecularData:
    """Clone a canonical graph and attach a ``pos`` ``[N+1, 3]`` float32 tensor.

    Prepends a dummy atom at index 0 (matching the upstream convention of
    "add_dummy_node=True") and stores the resulting 3-D coordinates on
    ``data.pos``.  Edge index and edge attributes are left untouched —
    the dummy stays disconnected.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; MAT inputs must be derived before batching"
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

    # Prepend the dummy node: zero vector with the first feature set to 1
    # so the encoder can recognise the dummy atom by inspection.  This
    # matches the upstream's ``m[0, 0] = 1`` convention.
    dummy_x = torch.zeros(1, x.shape[1], dtype=torch.float32, device=device)
    dummy_x[0, 0] = 1.0
    padded_x = torch.cat((dummy_x, x), dim=0)

    transformed = data.clone()
    transformed.x = padded_x
    transformed.pos = pos
    return transformed


def _embed_conformer(
    mol: Chem.Mol,
    *,
    sample: int | str,
    device: torch.device,
) -> Tensor:
    """Return ``[N+1, 3]`` float32 — real-atom coords prepended by the dummy."""

    # ``AllChem.EmbedMolecule`` and ``UFFOptimizeMolecule`` need explicit
    # hydrogens; we generate them, embed, optimise, then take the heavy-atom
    # positions (heavy-atom indices are preserved across ``AddHs``).
    mol_h = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol_h, maxAttempts=5000)
    if embed_status != 0:
        # Fall back to 2-D coords so the model still runs.  The paper does
        # not address this fallback explicitly; the upstream ``data_utils.py``
        # also uses ``Compute2DCoords`` on failure.
        AllChem.Compute2DCoords(mol_h)
    try:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    except RuntimeError:
        # Force-field setup can fail on exotic chemistries (e.g. charged
        # metals); fall through and use the unoptimised conformer.
        pass

    conf = mol_h.GetConformer()
    coords: list[tuple[float, float, float]] = []
    for heavy_index in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(heavy_index)
        point = conf.GetAtomPosition(atom.GetIdx())
        coords.append((float(point.x), float(point.y), float(point.z)))

    if len(coords) != mol.GetNumAtoms():
        raise TransformError(
            f"sample {sample} conformer produced {len(coords)} coords "
            f"for {mol.GetNumAtoms()} heavy atoms"
        )

    real_pos = torch.tensor(coords, dtype=torch.float32, device=device)
    dummy_pos = torch.full(
        (1, 3), _DUMMY_DISTANCE, dtype=torch.float32, device=device
    )
    pos = torch.cat((dummy_pos, real_pos), dim=0)
    if not torch.isfinite(pos).all():
        raise TransformError(
            f"sample {sample} conformer produced non-finite coordinates"
        )
    return pos


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_mat_inputs"]