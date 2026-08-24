# GraphMVP pretrained encoders

These are the encoder-only checkpoints from the supplied GraphMVP release.

- `GraphMVP_C.pth`: core Eq.7 checkpoint for the `simple` classification
  feature profile. It was selected from the `3D_hybrid_02_masking` output.
- `GraphMVP_G.pth`: core Eq.7 checkpoint for the `ogb_full` regression feature
  profile. It was selected from the `GraphMVP` regression output.

The `*_complete.pth`, `GraphMVP_CP`, `GraphMVP_AM`, and `3D_hybrid_03_masking`
artifacts were not copied: they contain pretraining-only 3-D/auxiliary heads
or Eq.8 extensions rather than the core downstream encoder contract.

Use the corresponding profile when passing a checkpoint to `GraphMVP`:

```text
feature_profile=simple  -> GraphMVP_C.pth
feature_profile=ogb_full -> GraphMVP_G.pth
```

`manifest.yaml` records the archive member, source revision, tensor profile,
and SHA-256 checksum. The project loader adapts the official 3-row chirality
table in `GraphMVP_C.pth` to the runtime's explicit 4-row canonical view.
