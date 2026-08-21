"""Small model-owned Mole-BERT MAM/TMCL training loop."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from .pretraining import MoleBERTPretrainer
from .tokenizer import MoleBERTTokenizer


def train_pretraining_epoch(
    pretrainer: MoleBERTPretrainer,
    tokenizer: MoleBERTTokenizer,
    batches: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Run one joint MAM/TMCL epoch and return mean loss."""

    pretrainer.train()
    tokenizer.eval()
    total = 0.0
    count = 0
    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = pretrainer(batch, tokenizer)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Mole-BERT pretraining produced a non-finite loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    if not count:
        raise ValueError("pretraining batches must not be empty")
    return total / count


def save_pretrained_encoder(pretrainer: MoleBERTPretrainer, path: str | Path) -> None:
    """Export the encoder state consumed by downstream Mole-BERT."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pretrainer.encoder.state_dict(), output)


__all__ = ["save_pretrained_encoder", "train_pretraining_epoch"]
