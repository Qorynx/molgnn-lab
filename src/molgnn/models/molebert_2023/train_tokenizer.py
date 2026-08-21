"""Small model-owned tokenizer training loop.

Dataset loading remains the caller's responsibility; this keeps unlabeled
ZINC/SMILES handling out of the shared supervised runner.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from .tokenizer import MoleBERTTokenizer, tokenizer_reconstruction_loss


def train_tokenizer_epoch(
    tokenizer: MoleBERTTokenizer,
    batches: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Run one VQ-VAE tokenizer epoch and return mean loss."""

    tokenizer.train()
    total = 0.0
    count = 0
    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = tokenizer_reconstruction_loss(tokenizer(batch), batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Mole-BERT tokenizer produced a non-finite loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    if not count:
        raise ValueError("tokenizer batches must not be empty")
    return total / count


def save_tokenizer(tokenizer: MoleBERTTokenizer, path: str | Path) -> None:
    """Save tokenizer, codebook, decoder, and architecture metadata."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": tokenizer.state_dict(),
            "num_layers": tokenizer.encoder.num_layers,
            "hidden_dim": tokenizer.encoder.hidden_dim,
            "num_tokens": tokenizer.quantizer.num_tokens,
            "group_boundaries": tokenizer.quantizer.group_boundaries,
        },
        output,
    )


__all__ = ["save_tokenizer", "train_tokenizer_epoch"]
