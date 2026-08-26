# Pretrain-GNNs official chemistry GIN checkpoints

This directory pins the four official molecular GIN pretraining checkpoints
from `snap-stanford/pretrain-gnns` (`chem/model_gin/*.pth`): ContextPred,
Attribute Masking, and their supervised-ChEMBL continuations. Each file is a
pure 57-tensor encoder state dict (5 layers, hidden 300, JK `last`).

The runtime loader in `molgnn.models.pretrain_gnns_2020.checkpoint` verifies
the manifest SHA-256, loads with `weights_only=True`, strictly checks keys and
shapes, and adapts the chirality embedding from the official 3 rows to the
runtime 4 rows (the new `other` row copies the `unspecified` row, which is
exactly how often `CHI_OTHER` appears in practice: never for standard RDKit
molecules). Scratch initialization stays the default; a variant is used only
when explicitly requested via `pretrained_variant` or an explicit checkpoint
path.

See `manifest.yaml` for provenance, feature schema, checksums, and license.
