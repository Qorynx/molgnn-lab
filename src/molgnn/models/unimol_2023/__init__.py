"""Uni-Mol (Zhou et al., ICLR 2023) supervised fine-tuning model.

Ports the dense 3-D Transformer backbone with additive pair-bias from
Gaussian RBF distances and a [CLS]-pooled head, on top of the lab's
canonical 153-dim atom features.

Documented deviations from the original Uni-Mol (per lab policy):

1. **Pretraining is not ported.**  Uni-Mol's headline contribution is
   self-supervised pretraining on 209 M PubChem 3-D conformers with
   ``MaskLMHead``, ``DistanceHead``, and an SE(3)-equivariant coordinate
   head.  Without pretraining, what we ship is a randomly-initialised
   dense Transformer with pair bias — *not* Uni-Mol in the sense the
   paper uses the term.  This is by far the largest reason our BACE
   ROC-AUC sits well below the paper's reported 0.85+.

2. **Atom-type vocabulary is collapsed.**  Upstream uses a 30-entry
   ``nn.Embedding`` for atom types *and* a 900-entry (30×30) ``edge_type``
   embedding that gives the pair bias per-pair-type context (C-C vs
   C-O bond, aromatic vs aliphatic, etc.).  The lab substitutes a
   continuous ``nn.Linear(153, embed_dim)`` and a single-edge-type
   Gaussian — see ``GaussianLayer`` in :mod:`.layers` for the
   consequence (the pair repr can no longer distinguish two bonds at
   the same distance with different chemistries).

3. **Bond features (14-dim) are not consumed.**  Uni-Mol builds its
   pair repr from distances alone; the lab transform still emits
   ``edge_index``/``edge_attr`` so PyG batching works, but the model
   reads only ``x``, ``pos``, ``batch``.

4. **Head output convention.**  Upstream fine-tune:
   ``ClassificationHead`` (tanh-pooler → dropout → ``Linear(inner,
   num_classes=2)``) paired with ``finetune_cross_entropy``.  Lab:
   ``LinearHead`` (``dropout → Linear(embed, num_targets=1)``) paired
   with ``bce_with_logits``.  Semantically the same (binary
   classification) but the loss contract dictates the output dim.

5. **3-D conformers are synthesised.**  Upstream trains on pre-computed
   PubChem 3-D conformers.  Lab: RDKit ``ETKDGv3`` +
   ``MMFFOptimizeMolecule`` via the shared geometry provider.

6. **Encoder dims are scaled down.**  Paper defaults
   ``embed_dim=512, ffn_embed_dim=2048 (4×), num_layers=15,
   num_heads=64, num_gaussian_kernels=128``; the lab smoke + BACE
   defaults (``embed_dim=64, ffn_embed_dim=128, num_layers=2,
   num_heads=2, num_gaussian_kernels=8``) preserve the architectural
   ratios (4× FFN, ``embed_dim % heads == 0``, ``K >= heads``) but
   shrink absolute capacity ~8-64× to fit CPU budget.

.. warning::

   BACE / BBBP-class downstream numbers are a lower bound on Uni-Mol's
   true capability for the reasons above (no pretraining, ETKDG
   conformers, smaller dims).  Numerical verification against the
   paper's equations is the right correctness check.
"""

from .model import UniMol

__all__ = ["UniMol"]