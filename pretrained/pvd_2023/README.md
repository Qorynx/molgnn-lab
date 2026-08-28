# Pre-training via Denoising checkpoint

`denoised-pcqm4mv2.weights.pt` is a safe weight-only conversion of the
official Lightning checkpoint bundled with
`shehzaidi/pre-training-via-denoising`.

The artifact contains the TorchMD-ET encoder, equivariant vector denoising
head, and running position-target normalizer. Optimizer, callback, scalar
property head, and Lightning runtime state were removed. Downstream
`pvd_torchmd_et` loads only the encoder and initializes its task head from
scratch; `PVDPretrainer` may restore all three retained components.

This checkpoint is for the official source profile (`sigma=0.04`, raw
i.i.d. noise, whitened vector normalization), not the paper's GNS-TAT model.

The source repository is MIT licensed. See `manifest.yaml` for provenance
and checksums.
