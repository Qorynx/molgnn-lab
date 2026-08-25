"""Uni-Mol (Zhou et al., ICLR 2023) supervised fine-tuning model.

Ports the dense 3-D Transformer backbone with additive pair-bias from
Gaussian RBF distances and a [CLS]-pooled head, on top of the lab's
canonical 153-dim atom features.  Pretraining (MaskLMHead, DistanceHead,
coordinate recovery, masked atom prediction on 209 M PubChem conformers)
is intentionally not ported per lab policy.

Caveat:  BACE / BBBP-class downstream numbers are a lower bound on
Uni-Mol's true capability because we synthesise 3-D conformers via
RDKit's ETKDG + MMFF, whereas the paper trained on pre-computed PubChem
3-D conformers.  Numerical verification against the paper's equations
is the right correctness check.
"""

from .model import UniMol

__all__ = ["UniMol"]