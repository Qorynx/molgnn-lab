# KPGT official pretrained checkpoint

`base.pth` is the KPGT-base checkpoint supplied in `pretrained.zip`. Its
tensor profile matches the official LiGhT base configuration: hidden size
768, 12 Transformer layers, 12 heads, path length 5, 512 fingerprint inputs,
and 200 molecular descriptors.

The project loads the 195 backbone tensors and intentionally skips the 12
pretraining-head tensors when constructing a downstream model. Scratch
initialization remains the default; select this artifact explicitly with:

```yaml
model:
  name: kpgt
  parameters:
    pretrained_checkpoint: pretrained/kpgt_2022/base.pth
```

See `manifest.yaml` for provenance, hashes, and the validated tensor profile.
