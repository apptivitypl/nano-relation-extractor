"""Evaluation loop shared by training and post-quantisation validation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..schema import LabelSchema
from .device import DeviceManager
from .losses import MultiTaskLoss
from .metrics import EvaluationResult, NerMetric, RelationMetric


class MultiTaskEvaluator:
    """Scores a model over an evaluation loader.

    Args:
        schema: Label vocabularies used to decode predictions.
        criterion: Loss module providing both the loss and the relation decoder.
        device_manager: Resolved device and mixed precision policy.
    """

    def __init__(
        self,
        schema: LabelSchema,
        criterion: MultiTaskLoss,
        device_manager: DeviceManager,
    ) -> None:
        self._schema = schema
        self._criterion = criterion
        self._device_manager = device_manager

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, loader: DataLoader) -> EvaluationResult:
        """Run the model over a loader and return both task scores.

        Args:
            model: Model to score. Left in evaluation mode on return.
            loader: Loader yielding :class:`MultiTaskBatch` objects.

        Returns:
            The combined evaluation result.
        """
        model.eval()
        device = self._device_manager.device
        ner_metric = NerMetric(self._schema)
        relation_metric = RelationMetric()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            with self._device_manager.autocast():
                outputs = model(**batch.model_inputs())
                losses = self._criterion(
                    ner_logits=outputs.ner_logits,
                    ner_labels=batch.ner_labels,
                    relation_logits=outputs.relation_logits,
                    relation_labels=batch.relation_labels,
                    pair_mask=batch.pair_mask,
                )
            total_loss += float(losses.total.item())
            num_batches += 1
            ner_metric.update(outputs.ner_logits.float(), batch.ner_labels)
            relation_metric.update(
                self._criterion.relation_objective.decode(
                    outputs.relation_logits.float()
                ),
                batch.relation_labels,
                batch.pair_mask,
            )

        return EvaluationResult(
            ner=ner_metric.compute(),
            relation=relation_metric.compute(),
            relation_recall_ceiling=relation_metric.recall_ceiling,
            loss=total_loss / num_batches if num_batches else None,
        )
