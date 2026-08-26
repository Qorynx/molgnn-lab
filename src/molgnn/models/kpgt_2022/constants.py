"""Source-faithful KPGT constants, vocabulary, and default profile.

Provenance:
- PAPER: "Knowledge-guided Pre-training of Graph Transformer for Molecular
  Property Prediction" (KDD 2022).
- OFFICIAL CODE: https://github.com/lihan97/KPGT.git revision
  ``47dc1646c70b2138a157de481d24a1ac35d174cd`` (files ``src/data/featurizer.py``,
  ``src/model/light.py``, and ``src/model_config.py``), Apache-2.0.

The vocabulary layout, sentinel values, and feature widths replicate the
official source so pretrained checkpoints stay loadable. Where the paper and
the official source differ, this implementation follows the official code
(two knowledge nodes, sparse path graph with ``path_length=5``, pre-LN,
hidden-dim attention scale, asymmetric virtual-path bias, GELU predictor).
"""

from __future__ import annotations

# --- Featurization widths (OFFICIAL CODE featurizer.py) ---------------------

D_NODE_FEATS = 137
D_EDGE_FEATS = 14
N_ATOM_TYPES = 101
N_BOND_TYPES = 5

# --- Sentinels (OFFICIAL CODE featurizer.py) --------------------------------

VIRTUAL_ATOM_INDICATOR = -1
VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1
# Official ``VIRTUAL_PATH_INDICATOR``; large negative so batching offsets and
# the model-side ``< -99`` clamp both leave it a padding marker.
VIRTUAL_PATH_INDICATOR = -1_000_000

# --- Knowledge node indicators (g.ndata['vavn'] semantics) ------------------
# 0: real bonded line-node; -1: isolated-atom node; 1: fingerprint node;
# 2: descriptor node.

FINGERPRINT_NODE_INDICATOR = 1
DESCRIPTOR_NODE_INDICATOR = 2

# --- Knowledge dimensions ---------------------------------------------------

FINGERPRINT_DIM = 512
DESCRIPTOR_DIM = 200

# --- Official 'base' profile (model_config.py) ------------------------------

KPGT_BASE_CONFIG: dict[str, float | int] = {
    "d_node_feats": D_NODE_FEATS,
    "d_edge_feats": D_EDGE_FEATS,
    "d_g_feats": 768,
    "d_hpath_ratio": 12,
    "n_mol_layers": 12,
    "path_length": 5,
    "n_heads": 12,
    "n_ffn_dense_layers": 2,
    "input_drop": 0.0,
    "attn_drop": 0.1,
    "feat_drop": 0.1,
}

DEFAULT_FINGERPRINT_RADIUS_ARGS = {"minPath": 1, "maxPath": 7}


class KPGTVocab:
    """Unordered (atom, bond, atom) triplet vocabulary from the official source.

    Layout (OFFICIAL CODE featurizer.py ``Vocab.construct``):

    - bonded triplets: for ``atom_id_1 <= atom_id_2`` over ``[0, 101)``, one id
      per bond type, ordered by atom1, then bond, then atom2;
    - isolated atoms: ``101`` extra ids after the bonded block;
    - one virtual id at the end.

    The total size is ``25857``.
    """

    def __init__(self, n_atom_types: int = N_ATOM_TYPES, n_bond_types: int = N_BOND_TYPES) -> None:
        if n_atom_types < 1 or n_bond_types < 1:
            raise ValueError("vocabulary sizes must be positive")
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self._vocab = self._construct()
        self.vocab_size = self._id

    def _construct(self) -> dict[int, dict[int, dict[int, int]]]:
        vocab: dict[int, dict[int, dict[int, int]]] = {}
        atom_ids = list(range(self.n_atom_types))
        bond_ids = list(range(self.n_bond_types))
        vocab_size = 0
        for atom_id_1 in atom_ids:
            vocab[atom_id_1] = {}
            for bond_id in bond_ids:
                vocab[atom_id_1][bond_id] = {}
                for atom_id_2 in atom_ids:
                    if atom_id_2 >= atom_id_1:
                        vocab[atom_id_1][bond_id][atom_id_2] = vocab_size
                        vocab_size += 1
        for atom_id in atom_ids:
            # Isolated-atom entries use the official 999 sentinels.
            vocab[atom_id][999] = {999: vocab_size}
            vocab_size += 1
        vocab[999] = {999: {999: vocab_size}}
        self._id = vocab_size + 1
        return vocab

    def index(self, atom_type1: int, atom_type2: int, bond_type: int) -> int:
        """Return the unordered triplet id, or ``vocab_size`` when unknown."""

        first, second = sorted((atom_type1, atom_type2))
        try:
            return self._vocab[first][bond_type][second]
        except KeyError:
            return self.vocab_size


def kpgt_vocab_size(n_atom_types: int = N_ATOM_TYPES, n_bond_types: int = N_BOND_TYPES) -> int:
    """Total vocabulary width without materializing the lookup tables."""

    bonded = (n_atom_types * (n_atom_types + 1) // 2) * n_bond_types
    return bonded + n_atom_types + 1


def kpgt_default_parameters() -> dict[str, object]:
    """Return the checkpoint-compatible downstream default parameters."""

    parameters: dict[str, object] = dict(KPGT_BASE_CONFIG)
    parameters.update(
        {
            "predictor_hidden_dim": 256,
            "predictor_num_layers": 2,
            "predictor_dropout": 0.0,
            "pretrained_checkpoint": None,
        }
    )
    return parameters


def attention_scale(hidden_dim: int) -> float:
    """Official query scale ``hidden_dim**-0.5`` (not the per-head variant)."""

    return hidden_dim**-0.5


__all__ = [
    "DEFAULT_FINGERPRINT_RADIUS_ARGS",
    "DESCRIPTOR_DIM",
    "DESCRIPTOR_NODE_INDICATOR",
    "D_EDGE_FEATS",
    "D_NODE_FEATS",
    "FINGERPRINT_DIM",
    "FINGERPRINT_NODE_INDICATOR",
    "KPGT_BASE_CONFIG",
    "N_ATOM_TYPES",
    "N_BOND_TYPES",
    "VIRTUAL_ATOM_FEATURE_PLACEHOLDER",
    "VIRTUAL_ATOM_INDICATOR",
    "VIRTUAL_BOND_FEATURE_PLACEHOLDER",
    "VIRTUAL_PATH_INDICATOR",
    "KPGTVocab",
    "attention_scale",
    "kpgt_default_parameters",
    "kpgt_vocab_size",
]
