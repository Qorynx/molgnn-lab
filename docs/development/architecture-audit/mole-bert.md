# Mole-BERT architecture audit and implementation plan

## Scope and conclusion

This record audits Xia et al., *Mole-BERT: Rethinking Pre-training Graph
Neural Networks for Molecules* (ICLR 2023), against the supplied author
repository snapshot.

Implementation is feasible in the current project with a model-local 2-D
input transform. Mole-BERT uses molecular connectivity, atom identity,
chirality, bond type, and bond direction; it does not use coordinates,
distances, conformers, or equivariance. Consequently:

- all 12 current MoleculeNet schemas can use the same downstream model;
- QM9 can use its 2-D covalent graph while `pos` is retained but ignored;
- no geometry contract or change to the shared featurizer is needed; and
- pretraining is **essential** to the Mole-BERT claim. A randomly initialized
  downstream GIN is a useful compatibility baseline, but is not a reproduced
  Mole-BERT result.

The requested milestone must therefore include the tokenizer, self-supervised
MAM + TMCL pretrainer, checkpoint contract, and supervised downstream model.
It is not complete if it ports only the GIN fine-tuning wrapper.

## Evidence and provenance

Evidence priority:

1. `papers/mole-bert.pdf`, the ICLR 2023 paper.
2. `E:/VsCodeProject/legacy code/Mole-BERT`, author repository
   `junxia97/Mole-BERT`, commit
   `2feff8a33e3634b66b7408e2e2780fc9d960909f`.
3. The repository's scripts, dataset conversion, and released encoder state.
4. Current `molgnn-lab` 2-D data, registry, task, and smoke contracts.

The repository has no license file or license declaration: redistribution
terms are `UNKNOWN`. Use it as behavioral evidence and independently
implement the equations. On 2026-08-21, the project owner explicitly directed
that the supplied encoder checkpoint be committed despite this uncertainty.
That repository decision is recorded with the artifact; it does not establish
upstream redistribution rights.

The local released state is an encoder-only 5-layer/300-dimensional GIN
checkpoint at `model_gin/Mole-BERT.pth`, SHA-256
`375CD40AF9F21D2A92ED1ACBDEA9EFAD14254C36703BB0E3A7E433E09E624CE1`.
It contains neither the tokenizer/codebook nor MAM/TMCL heads. Its shapes are
useful for testing an opt-in legacy checkpoint converter.

## Method that must be preserved

### Context-aware tokenizer

The paper first trains a group VQ-VAE on two million unlabeled ZINC15
molecules. A 5-layer GIN encodes every atom context, a 512-entry codebook
quantizes the atom embedding within an element group, and a GNN decoder
reconstructs molecular attributes. The paper uses commitment coefficient
`beta=0.25`; its reference schedule is 60 epochs, batch size 256, and learning
rate 0.001.

Carbon, nitrogen, oxygen, and all remaining elements must use disjoint
codebook regions. Codebook boundaries belong in the tokenizer checkpoint
manifest; they must not be inferred during later pretraining.

### Mole-BERT pretraining

The downstream encoder is another bond-aware GIN. For every clean graph:

1. create two independently masked views, normally at rates 0.15 and 0.30;
2. use the frozen tokenizer **and its VQ codebook** to obtain 512-way labels;
3. apply MAM to predict the masked atom tokens;
4. mean-pool and project the two graph representations;
5. apply contrastive loss with temperature `tau=0.1`; and
6. enforce that the light-mask view is closer to the clean graph than the
   heavy-mask view with the paper's cosine-similarity triplet hinge.

The total paper objective is `L_MAM + L_con + mu * L_tri`; `mu` is selected
from `{0.1, 0.3, 0.5}`. The reference pretraining profile is 5 GIN layers,
hidden width 300, 100 epochs, batch size 256, and learning rate 0.001.

Only the pretrained GIN encoder is transferred. Projection, masking, token
classification, and tokenizer modules are not used by the normal downstream
forward pass.

### Downstream model

The paper profile is a 5-layer, 300-dimensional edge-aware GIN with BatchNorm,
ReLU/dropout on all but the last layer, last-layer JK, mean graph pooling, and
a task head. Each message is formed from the source-node embedding plus bond
type and bond-direction embeddings; a learned self-loop bond type is injected
inside every layer.

The project wrapper must return raw `[num_graphs, num_targets]` values:
regression values for MSE/MAE and logits for masked BCE-with-logits. It must
not hard-code dataset names, target widths, or label conventions.

## Paper/source discrepancies and decisions

| Topic | PAPER | OFFICIAL CODE | PROJECT DECISION |
| --- | --- | --- | --- |
| Codebook groups | Gives an equal 128/128/128/128 example | Uses abundance-skewed Python slices and accidentally leaves IDs 377, 433, 488, and 511 unreachable | Use contiguous, fully covered groups stored in the manifest; paper-equal groups are the initial default. Record any later source-compatible profile explicitly. |
| Token labels | Nearest entry in the frozen group VQ codebook | Loads the codebook but never uses it; `argmax` is applied to a 300-wide encoder output, so labels cannot span 512 tokens | Implement the paper VQ lookup; do not reproduce this defect. |
| Tokenizer script | 5-layer GIN encoder/decoder and VQ loss | Constructor call is inconsistent, and categorical CE replaces the stated scaled-cosine reconstruction term | Implement the paper objective; add a tiny executable tokenizer-training test before long ZINC training. |
| Triplet loss | Cosine-similarity hinge | `TripletMarginLoss(margin=0)` over unnormalized Euclidean representations | Use the paper cosine hinge. |
| Bond masking | Not part of the defined MAM objective | Adds a default 4-way masked-bond auxiliary loss | Keep it as an optional, clearly named source auxiliary; it is off in the paper profile. |
| Chirality | Four tags, including `other` | Loader emits four IDs but encoder embedding has only three rows | Use four rows. Legacy import copies rows 0..2 and initializes `other` from `unspecified`, reporting the adaptation. |
| Fine-tuning coverage | Eight MoleculeNet classification sets; also reports ESOL and Lipo regression | Script supports nine classification sets including PCBA, but no regression | Use the shared task/target-width contract for all 12 current schemas. |
| QM9 | Only 30,000 carbon atoms are sampled for tokenizer visualization | No QM9 training path | Support QM9 regression as a project extension; do not claim it was a paper benchmark. |

Where paper equations and executable behavior conflict, the initial profile
follows the paper. Source-only behavior must never be introduced silently.

## Project data contract

Add a model-specific transform `molebert_inputs`; do not modify the canonical
153-dimensional atom or 14-dimensional bond schema. It appends:

```text
molebert_atom_attr   LongTensor[N, 2]  # atomic-number index, chirality
molebert_bond_attr   LongTensor[E, 2]  # bond type, bond direction
```

Atomic number, chirality, and bond type are decoded exactly from the canonical
one-hot blocks. Bond direction is absent from the canonical schema, so the
transform obtains it from the existing `data.smiles` with RDKit and aligns the
parsed graph to the canonical atom/edge order. Alignment must validate atom
count, atomic numbers, connectivity, and bond types. Ambiguous mappings that
produce different directional features must fail rather than silently use a
wrong direction.

This path is exact for the CSV-SMILES datasets because their canonical graph
comes from the same RDKit molecule. The QM9 adapter also supplies SMILES; use
a graph-isomorphism alignment because its SDF atom order need not equal
canonical SMILES order. QM9 geometry and explicit hydrogen atoms remain
untouched. No `MolecularData.__inc__` override is needed because the new
fields contain attributes, not indices.

Pretraining masking is performed in a Mole-BERT-owned collator on cloned
samples. It must never mutate cached dataset objects. Masked node/bond IDs and
view membership are ephemeral pretraining data, not shared dataset fields.

## Proposed module and artifact layout

```text
src/molgnn/models/molebert_2023/
    __init__.py
    layers.py          # discrete bond-aware GIN
    tokenizer.py       # group VQ-VAE and token lookup
    pretraining.py     # masking, MAM, TMCL, losses
    checkpoint.py      # native manifest and legacy encoder conversion
    model.py           # downstream graph predictor
    train_tokenizer.py # python -m entry point
    pretrain.py        # python -m entry point
src/molgnn/transforms/molebert.py
pretrained/molebert_2023/
    README.md
    manifest.example.yaml
```

Training/checkpoint commands stay model-owned so the shared supervised
trainer does not gain `if model == "molebert"` branches. Real weights live
outside `src/molgnn` and are not bundled in the wheel. The manifest records
feature schema, group boundaries, architecture, protocol, source dataset,
seed, software versions, state checksum, and whether the state was trained by
this project or converted.

Register one downstream runtime, `molebert`, with `molebert_inputs`, no
geometry requirement, and an optional checkpoint path. A missing checkpoint
must be reported as `initialization=scratch`; loading the official/local or a
project-trained state must be reported as `initialization=pretrained`. Do not
create two architecture aliases that differ only by initialization.

The supplied encoder state and provenance manifest are stored in
`pretrained/molebert_2023/` by explicit project-owner decision. The first
implementation should load this pinned artifact through the same converter
used for user-supplied states. Independently trained checkpoints follow the
normal artifact policy and must use distinct names/manifests.

## Minimal implementation sequence

1. **MB-01 — discrete input transform.** Implement exact canonical-feature
   decoding, SMILES/SDF graph alignment, bond direction, validation, empty
   bonds, and deterministic batching.
2. **MB-02 — encoder and downstream head.** Implement the paper/source GIN,
   four-tag chirality fix, JK modes, mean pooling, `[B,T]` head, and strict
   batch validation. Add `molebert` registration and a small smoke profile.
3. **MB-03 — checkpoint contract.** Add versioned manifests, safe
   `weights_only` loading, architecture/schema checks, and the explicit
   3-to-4 chirality legacy conversion. Test parity for source-supported IDs.
4. **MB-04 — tokenizer.** Implement group VQ lookup, straight-through
   quantization, reconstruction decoder/loss, frozen token extraction, and
   tokenizer checkpointing. Verify every one of 512 IDs is reachable by its
   declared group.
5. **MB-05 — MAM + TMCL.** Implement independent 0.15/0.30 masks, clean
   anchor, projection head, 512-way MAM, contrastive loss, cosine triplet
   hinge, optional source bond auxiliary, and encoder-only export.
6. **MB-06 — model-owned training entry points.** Add short-resume-safe
   tokenizer/pretraining commands using unlabeled 2-D ZINC records; keep
   optimizer/schedule configuration outside the runtime architecture.
7. **MB-07 — integration.** Run the downstream lifecycle on all 12
   MoleculeNet schemas and a native-QM9 subset, once from scratch for runtime
   isolation and once from a small project-pretrained checkpoint for transfer
   validation.
8. **MB-08 — full pretraining, optional expensive run.** Only after the
   functional gates pass, train on the paper's 2M-ZINC/60+100 epoch schedule.
   This is required for a paper-scale checkpoint, not for the core code gate.

## Required verification

- transform values match direct RDKit paper features, including stereo bond
  direction, and remain aligned after PyG batching;
- every GIN message, self-loop type, BatchNorm/dropout schedule, JK readout,
  and mean pool matches an independent small reference;
- legacy checkpoint conversion reports the chirality adaptation and produces
  matching encoder outputs for IDs 0..2;
- group masks are disjoint and exhaustive; token lookup demonstrably uses the
  VQ codebook and can return IDs above 299;
- two masked views do not mutate the clean graph or each other;
- MAM, contrastive, cosine triplet, and combined loss match hand-computed tiny
  cases; tokenizer parameters receive no pretraining gradients;
- forward, masked loss, backward, one optimizer step, state round-trip, and
  resume work for both classification and regression;
- outputs support missing labels and real target widths, including ToxCast
  617 and PCBA 128;
- `molgnn describe-model --model molebert` declares only the model-local 2-D
  fields and no geometry;
- `molgnn moleculenet-smoke --models molebert --epochs 1 --device cpu`
  reports 12 passed and 0 failed; and
- a small QM9 regression run completes through the shared runner while
  changing or removing `pos` does not affect predictions.

## Boundaries and non-claims

- QM9 property prediction is supported by project design, but the paper uses
  QM9 only for tokenizer visualization; no Mole-BERT QM9 score is available
  for parity.
- FreeSolv and PCBA are compatibility extensions beyond the paper's reported
  downstream table (PCBA exists in the source fine-tuner).
- A scratch run tests architecture/runtime compatibility only; it is not a
  Mole-BERT pretraining result.
- A tiny pretraining smoke proves lifecycle correctness, not representation
  quality. Paper-scale claims require the 2M-ZINC schedule and documented
  splits/seeds.
- The supplied official encoder is committed by project-owner decision, while
  its upstream redistribution/license status remains `UNKNOWN` and must stay
  visible in its manifest.
- Protein/DTA heads, Malaria, CEP, Davis, and KIBA are outside this milestone;
  they require datasets/task adapters not present in the requested
  MoleculeNet + QM9 scope.

## Definition of done

The implementation is complete when the paper-correct tokenizer and MAM +
TMCL pretrainer export a versioned encoder checkpoint; the downstream
`molebert` runtime can explicitly report scratch versus pretrained
initialization; focused architecture/pretraining/checkpoint tests pass; all 12
MoleculeNet smoke runs pass; and a native-QM9 regression subset trains while
the model ignores `pos`. Exact paper-score reproduction and the full 2M-ZINC
run remain separate, expensive experiment milestones.
