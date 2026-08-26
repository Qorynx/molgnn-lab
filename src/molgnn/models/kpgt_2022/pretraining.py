"""KPGT knowledge-guided pretraining objectives and tiny training helpers.

Provenance: ``OFFICIAL CODE`` revision ``47dc1646c70b2138a157de481d24a1ac35d174cd``
(``src/data/collator.py``, ``src/trainer/pretrain_trainer.py``,
``scripts/train_kpgt.py``). Contract replicated:

- vocabulary of 25,857 unordered line-node token types;
- 50% of real/isolated line-nodes selected balanced per token type;
- corruption 80% mask-token / 10% random-token (different type) / rest kept,
  with feature pre-replacement for replaced tokens;
- 50% fingerprint bits flipped and 50% descriptor values overwritten with
  Uniform(0,1) using the official base rates;
- predictions cover the full fingerprint/descriptor vectors and every
  selected token; loss ``(token CE + fp BCE-with-logits + md MSE) / 3``;
- gradient-norm clipping at 5.

The corruption RNG differs from NumPy streams of the original environment
(framework parity is impossible); sampling *semantics* follow the source.
This module never touches the shared Trainer lifecycle.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .constants import kpgt_vocab_size
from .layers import init_params
from .model import KPGT


def balanced_candidate_probabilities(labels: Tensor) -> Tensor:
    """Per-node probabilities uniform within each token-type group."""

    counts = torch.bincount(labels)
    weights = 1.0 / counts[labels].to(torch.float64)
    return weights / weights.sum()


def _choice_without_replacement(
    population: int, size: int, weights: Tensor | None, generator: torch.Generator | None
) -> Tensor:
    if size <= 0:
        return torch.empty(0, dtype=torch.long)
    if weights is None:
        return torch.randperm(population, generator=generator)[:size]
    return torch.multinomial(weights, size, replacement=False, generator=generator)


def mask_line_nodes(
    indicators: Tensor,
    labels: Tensor,
    *,
    candi_rate: float = 0.5,
    mask_rate: float = 0.8,
    replace_rate: float = 0.1,
    keep_rate: float = 0.1,
    generator: torch.Generator | None = None,
    begin_end: Tensor | None = None,
    bond_attr: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Official BERT-style corruption over batched line-nodes.

    Returns ``(mask, sl_labels, begin_end, bond_attr, indicators)`` where
    mask values follow the official encoding (1 masked, 2 replaced, 3 kept,
    else untouched), ``sl_labels`` holds the original token ids of every
    selected node, and the feature/indicator tensors are replaced clones when
    pre-replacement applies (official copies ``begin_end``/``edge``/``vavn``
    of a random differently-labelled valid node).
    """

    device = indicators.device
    valid_positions = torch.nonzero(indicators <= 0, as_tuple=False).flatten()
    valid_labels_cpu = labels.index_select(0, valid_positions).cpu()
    probabilities = balanced_candidate_probabilities(valid_labels_cpu)

    num_valid = int(valid_positions.numel())
    num_candidates = int(num_valid * candi_rate)
    candidate_local = _choice_without_replacement(
        num_valid, num_candidates, probabilities, generator
    )
    permuted = candidate_local[torch.randperm(len(candidate_local), generator=generator)]
    mask_count = int(len(candidate_local) * mask_rate)
    mask_local = permuted[:mask_count]
    remaining_local = _setdiff(candidate_local, mask_local)
    replace_fraction = replace_rate / (1.0 - keep_rate)
    replace_count = int(len(remaining_local) * replace_fraction)
    shuffled_remaining = remaining_local[
        torch.randperm(len(remaining_local), generator=generator)
    ]
    replace_local = shuffled_remaining[:replace_count]
    keep_local = _setdiff(remaining_local, replace_local)

    mask = torch.zeros(int(indicators.shape[0]), dtype=torch.long, device=device)
    selected_global = valid_positions[
        torch.cat((mask_local, replace_local, keep_local)).to(device)
    ]
    values = torch.cat(
        (
            torch.ones(len(mask_local), dtype=torch.long),
            torch.full((len(replace_local),), 2, dtype=torch.long),
            torch.full((len(keep_local),), 3, dtype=torch.long),
        )
    ).to(device)
    mask[selected_global] = values

    # Boolean indexing preserves the same node order used by
    # ``states[mask >= 1]`` in the prediction head (OFFICIAL CODE line 57).
    sl_labels = labels[mask >= 1]

    updated_begin_end = begin_end
    updated_bond_attr = bond_attr
    updated_indicators = indicators
    if begin_end is not None and bond_attr is not None and len(replace_local) > 0:
        replace_ids = valid_positions[replace_local.to(device)]
        replace_labels_cpu = labels.index_select(0, replace_ids).cpu()
        if int(torch.unique(valid_labels_cpu).numel()) < 2:
            # A different token type does not exist in this batch. Treat the
            # selected replacements as kept tokens instead of rejection-looping.
            mask[replace_ids] = 3
        else:
            source_local = torch.multinomial(
                probabilities, len(replace_ids), replacement=True, generator=generator
            )
            source_labels_cpu = valid_labels_cpu[source_local]
            # OFFICIAL CODE resamples source node ids until their labels differ.
            for _ in range(64):
                unequal = source_labels_cpu != replace_labels_cpu
                if bool(unequal.all()):
                    break
                todo = torch.nonzero(~unequal, as_tuple=False).flatten()
                fresh = torch.multinomial(
                    probabilities, len(todo), replacement=True, generator=generator
                )
                source_local[todo] = fresh
                source_labels_cpu[todo] = valid_labels_cpu[fresh]
            if not bool((source_labels_cpu != replace_labels_cpu).all()):
                raise RuntimeError("failed to sample a different KPGT token type")
            new_ids = valid_positions[source_local.to(device)]
            updated_begin_end = begin_end.clone()
            updated_bond_attr = bond_attr.clone()
            updated_indicators = indicators.clone()
            updated_begin_end[replace_ids] = updated_begin_end[new_ids]
            updated_bond_attr[replace_ids] = updated_bond_attr[new_ids]
            # OFFICIAL CODE also copies vavn; replacements stay real/isolated
            # because candidates are drawn from vavn <= 0 nodes only.
            updated_indicators[replace_ids] = updated_indicators[new_ids]

    return mask, sl_labels, updated_begin_end, updated_bond_attr, updated_indicators


def _setdiff(universe: Tensor, remove: Tensor) -> Tensor:
    """Order-preserving complement (np.setdiff1d without sorting)."""

    remove_set = set(remove.tolist())
    keep = [index for index in universe.tolist() if index not in remove_set]
    return torch.tensor(keep, dtype=torch.long)


def disturb_fingerprint(
    fingerprint: Tensor, rate: float = 0.5, generator: torch.Generator | None = None
) -> Tensor:
    """Flip exactly ``int(B*D*rate)`` random bits over the flat batch."""

    disturbed = fingerprint.detach().clone()
    flattened = disturbed.reshape(-1)
    count = int(flattened.numel() * rate)
    if count > 0:
        positions = torch.randperm(flattened.numel(), generator=generator)[:count]
        flattened[positions] = 1.0 - flattened[positions]
    return disturbed.reshape(fingerprint.shape)


def disturb_descriptor(
    descriptor: Tensor, rate: float = 0.5, generator: torch.Generator | None = None
) -> Tensor:
    """Overwrite ``int(B*D*rate)`` descriptor entries with Uniform(0,1)."""

    disturbed = descriptor.detach().clone()
    flattened = disturbed.reshape(-1)
    count = int(flattened.numel() * rate)
    if count > 0:
        positions = torch.randperm(flattened.numel(), generator=generator)[:count]
        noise = torch.rand(count, generator=generator).to(flattened.device)
        flattened[positions] = noise
    return disturbed.reshape(descriptor.shape)


class KPGTPretrainer(nn.Module):
    """Backbone plus the three official knowledge-guided prediction heads."""

    def __init__(
        self,
        backbone: KPGT | None = None,
        *,
        n_node_types: int | None = None,
        **backbone_kwargs: object,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone_kwargs.setdefault("num_targets", 1)
            backbone = KPGT(**backbone_kwargs)  # type: ignore[arg-type]
        self.backbone = backbone
        d_g_feats = backbone.d_g_feats
        if n_node_types is None:
            n_node_types = kpgt_vocab_size()
        self.n_node_types = int(n_node_types)
        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            nn.GELU(),
            nn.Linear(d_g_feats, self.n_node_types),
        )
        d_fp_feats = int(backbone.triplet_emb.fp_proj.in_proj.in_features)
        d_md_feats = int(backbone.triplet_emb.md_proj.in_proj.in_features)
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), nn.GELU(), nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), nn.GELU(), nn.Linear(d_g_feats, d_md_feats)
        )
        for module in (self.node_predictor, self.fp_predictor, self.md_predictor):
            module.apply(init_params)

    def forward(
        self,
        fields: dict[str, Tensor],
        disturbed_fingerprint: Tensor,
        disturbed_descriptor: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        backbone = self.backbone
        indicators = fields["kpgt_node_indicator"]
        token_count = fields["kpgt_token_count"]
        total_nodes = int(indicators.shape[0])
        node_graph_ids, _ = backbone.node_layout(token_count, total_nodes)

        knowledge_fields = dict(fields)
        knowledge_fields["kpgt_fingerprint"] = disturbed_fingerprint
        knowledge_fields["kpgt_descriptor"] = disturbed_descriptor
        fp_nodes, md_nodes = backbone.project_knowledge(knowledge_fields, node_graph_ids)
        triplet_h = backbone.embed_triplet_states(fields, fp_nodes, md_nodes)
        masked = mask == 1
        if bool(masked.any()):
            triplet_h = triplet_h.clone()
            triplet_h[masked] = backbone.mask_emb.weight[0]
        states = backbone.model(
            triplet_h,
            fields["kpgt_attention_edge_index"],
            fields["kpgt_path_index"],
            fields["kpgt_virtual_path"],
            fields["kpgt_self_loop"],
        )
        sl_predictions = self.node_predictor(states[mask >= 1])
        fp_predictions = self.fp_predictor(states[indicators == 1])
        md_predictions = self.md_predictor(states[indicators == 2])
        return sl_predictions, fp_predictions, md_predictions


def compute_pretraining_losses(
    sl_predictions: Tensor,
    fp_predictions: Tensor,
    md_predictions: Tensor,
    sl_labels: Tensor,
    fingerprint: Tensor,
    descriptor: Tensor,
    *,
    fp_pos_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    """Official three-objective loss averaged into one scalar."""

    sl_loss = F.cross_entropy(sl_predictions, sl_labels, reduction="mean")
    fp_loss = F.binary_cross_entropy_with_logits(
        fp_predictions, fingerprint, weight=fp_pos_weight, reduction="mean"
    )
    md_loss = F.mse_loss(md_predictions, descriptor, reduction="mean")
    total = (sl_loss + fp_loss + md_loss) / 3
    return {
        "loss": total,
        "sl_loss": sl_loss,
        "fp_loss": fp_loss,
        "md_loss": md_loss,
    }


def corrupt_pretraining_batch(
    batch: object,
    *,
    candi_rate: float = 0.5,
    fp_disturb_rate: float = 0.5,
    md_disturb_rate: float = 0.5,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Apply official corruption to one already-batched sample collection."""

    mask, sl_labels, begin_end, bond_attr, indicators = mask_line_nodes(
        batch.kpgt_node_indicator,
        batch.kpgt_triplet_label,
        candi_rate=candi_rate,
        generator=generator,
        begin_end=batch.kpgt_begin_end,
        bond_attr=batch.kpgt_bond_attr,
    )
    fingerprint = batch.kpgt_fingerprint
    descriptor = batch.kpgt_descriptor
    return {
        "mask": mask,
        "sl_labels": sl_labels,
        "begin_end": begin_end,
        "bond_attr": bond_attr,
        "indicators": indicators,
        "fingerprint": disturb_fingerprint(fingerprint, fp_disturb_rate, generator),
        "descriptor": disturb_descriptor(descriptor, md_disturb_rate, generator),
        "target_fingerprint": fingerprint,
        "target_descriptor": descriptor,
    }


def train_pretraining_epoch(
    pretrainer: KPGTPretrainer,
    batches: list[object],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    *,
    fp_pos_weight: Tensor | None = None,
    grad_clip: float = 5.0,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """One tiny pretraining epoch over an explicit list of batched samples."""

    pretrainer.train()
    totals = {"loss": 0.0, "sl_loss": 0.0, "fp_loss": 0.0, "md_loss": 0.0}
    count = 0
    target_device = torch.device(device)
    for batch_index, batch in enumerate(batches):
        corrupted = corrupt_pretraining_batch(batch, generator=generator)
        corrupted_features = {
            "kpgt_begin_end": corrupted["begin_end"],
            "kpgt_bond_attr": corrupted["bond_attr"],
            "kpgt_node_indicator": corrupted["indicators"],
        }
        fields = {
            name: (
                corrupted_features[name].to(target_device)
                if name in corrupted_features
                else getattr(batch, name).to(target_device)
            )
            for name in pretrainer.backbone.required_batch_fields
        }
        optimizer.zero_grad(set_to_none=True)
        sl_predictions, fp_predictions, md_predictions = pretrainer(
            fields,
            corrupted["fingerprint"].to(target_device),
            corrupted["descriptor"].to(target_device),
            corrupted["mask"].to(target_device),
        )
        losses = compute_pretraining_losses(
            sl_predictions,
            fp_predictions,
            md_predictions,
            corrupted["sl_labels"].to(target_device),
            corrupted["target_fingerprint"].to(target_device),
            corrupted["target_descriptor"].to(target_device),
            fp_pos_weight=(
                fp_pos_weight.to(target_device) if fp_pos_weight is not None else None
            ),
        )
        losses["loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(pretrainer.parameters(), grad_clip)
        optimizer.step()
        for name in totals:
            totals[name] += float(losses[name].item())
        count += 1
    if count == 0:
        raise ValueError("pretraining epoch requires at least one batch")
    return {name: value / count for name, value in totals.items()}


def save_pretraining_checkpoint(
    path: str | Path,
    pretrainer: KPGTPretrainer,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> Path:
    """Persist encoder/backbone state (plus optional optimizer) portably."""

    payload: dict[str, object] = {
        "format_version": 1,
        "step": int(step),
        "model": pretrainer.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def resume_pretraining_checkpoint(
    path: str | Path,
    pretrainer: KPGTPretrainer,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Restore a tiny pretraining run saved by :func:`save_pretraining_checkpoint`."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"not a KPGT pretraining checkpoint: {path}")
    pretrainer.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {"step": int(payload.get("step", 0))}


__all__ = [
    "KPGTPretrainer",
    "balanced_candidate_probabilities",
    "compute_pretraining_losses",
    "corrupt_pretraining_batch",
    "disturb_descriptor",
    "disturb_fingerprint",
    "mask_line_nodes",
    "resume_pretraining_checkpoint",
    "save_pretraining_checkpoint",
    "train_pretraining_epoch",
]
