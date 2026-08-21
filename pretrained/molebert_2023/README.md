# Mole-BERT pretrained encoder

`Mole-BERT.pth` is the encoder-only state distributed in the supplied author
repository snapshot. It does not contain the VQ tokenizer, codebook, MAM
heads, TMCL projection head, optimizer, or training configuration.

The upstream repository contains no license declaration. The checkpoint was
committed by explicit project-owner decision on 2026-08-21. That decision is
recorded here for provenance and does not establish upstream redistribution
rights.

See `manifest.yaml` for the pinned source revision, tensor profile, and
checksum. Runtime loading/conversion will be added with the Mole-BERT model;
until then this directory is an inert artifact.
