"""Architecture-only base contract for molecular graph models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn
from torch_geometric.data import Batch


@dataclass(frozen=True)
class ModelTrainingOutput:
    """Primary prediction plus optional architecture-owned auxiliary heads."""

    prediction: Tensor
    auxiliary_predictions: tuple[Tensor, ...] = ()


class BaseMolecularModel(nn.Module, ABC):
    """Minimal model boundary shared by every architecture implementation.

    Training, loss calculation, optimization, and configuration parsing stay
    outside this class so architectures can be swapped without changing the
    shared experiment lifecycle.
    """

    @abstractmethod
    def forward(self, batch: Batch) -> Tensor:
        """Return one ``[batch_size, num_targets]`` tensor per graph."""

    def forward_training(self, batch: Batch) -> ModelTrainingOutput:
        """Return the standard prediction when no auxiliary objective exists."""

        return ModelTrainingOutput(self(batch))


__all__ = ["BaseMolecularModel", "ModelTrainingOutput"]
