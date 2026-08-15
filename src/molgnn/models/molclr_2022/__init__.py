"""MolCLR (Wang et al., 2022) supervised fine-tuning models.

Ports the GIN- and GCN-based encoders of MolCLR with the lab's
canonical 153-dim atom / 14-dim bond featurization.  Pretraining,
contrastive losses, and graph augmentations (atom masking, bond
deletion, subgraph removal) are intentionally not ported; this
module provides drop-in replacements for the supervised fine-tune
backbones used in Tables 1 and 2 of the paper.
"""

from .model import MolCLRGIN, MolCLRGCN

__all__ = ["MolCLRGIN", "MolCLRGCN"]
