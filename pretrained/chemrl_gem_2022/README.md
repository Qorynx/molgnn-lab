# ChemRL-GEM official pretrained encoders

This directory pins the two encoder checkpoints linked by the official
PaddleHelix ChemRL-GEM README. `class.pdparams` is the classification variant
and `regr.pdparams` is the regression variant. They have identical key/shape
schemas (164 tensors) but different learned values and must not be merged or
silently interchanged.

The files remain in their original Paddle format for provenance. They are not
yet runtime-ready in molgnn-lab: implementation must add a strict, allowlisted
Paddle-to-PyTorch converter, transpose only Paddle linear kernels, verify all
keys/shapes, and record the converted hashes. Smoke tests should default to
scratch initialization; users select `classification` or `regression`
explicitly when pretrained initialization is desired.

See `manifest.yaml` for the official URL, revisions, checksums, scope, safety
notes, and unresolved license status.

