"""Loss objectives for the multi-task model.

The relation objective owns both the loss and the decision rule, because the two
are inseparable: adaptive thresholding compares each class against a learned
per-pair threshold, whereas a binary cross entropy objective compares against a
fixed probability. Bundling them prevents a mismatched pair from silently
producing meaningless predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@dataclass
class MultiTaskLossOutput:
    """Loss terms produced for a single batch.

    Attributes:
        total: Weighted sum optimised by the trainer.
        ner: Unweighted token classification loss.
        relation: Unweighted relation classification loss.
    """

    total: torch.Tensor
    ner: torch.Tensor
    relation: torch.Tensor


@runtime_checkable
class RelationObjective(Protocol):
    """Pairs a relation loss with the decision rule it implies."""

    @property
    def name(self) -> str:
        """Identifier written into training reports and the model card."""
        ...

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the relation loss over unpadded pairs.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.
            targets: Multi-hot targets, shape ``[B, P, R]``.
            pair_mask: Validity flag per pair, shape ``[B, P]``.

        Returns:
            A scalar loss tensor.
        """
        ...

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert relation scores into binary predictions.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.

        Returns:
            A boolean tensor of shape ``[B, P, R]``. Column zero is always
            ``False`` because ``NA`` is the absence of a prediction.
        """
        ...

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert relation scores into per-class confidences.

        The confidence must be consistent with :meth:`decode`: a class predicted
        by ``decode`` scores above ``0.5`` here, and one rejected scores below.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.

        Returns:
            A float tensor of shape ``[B, P, R]`` with values in ``(0, 1)``.
        """
        ...


class AdaptiveThresholdLoss(nn.Module):
    """Adaptive thresholding objective for multi-label relation extraction.

    Class zero acts as a per-pair threshold that the network learns jointly with
    the relation scores. Positive classes are ranked above it and negative
    classes below it, which removes the global probability threshold that a
    plain binary objective needs and that a three percent positive rate makes
    hard to tune.

    The formulation follows Zhou et al., "Document-Level Relation Extraction with
    Adaptive Thresholding and Localized Context Pooling" (AAAI 2021).
    """

    @property
    def name(self) -> str:
        """Identifier written into training reports and the model card."""
        return "adaptive_threshold"

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the adaptive thresholding loss.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.
            targets: Multi-hot targets with ``NA`` in column zero, ``[B, P, R]``.
            pair_mask: Validity flag per pair, shape ``[B, P]``.

        Returns:
            A scalar loss tensor averaged over unpadded pairs.
        """
        selected_logits, selected_targets = _select_valid_pairs(
            logits, targets, pair_mask
        )
        if selected_logits.numel() == 0:
            return logits.sum() * 0.0

        positives = selected_targets.clone()
        positives[:, 0] = 0.0
        threshold = torch.zeros_like(selected_targets)
        threshold[:, 0] = 1.0

        floor = torch.finfo(selected_logits.dtype).min

        positive_mask = positives + threshold
        positive_logits = selected_logits.masked_fill(positive_mask == 0, floor)
        positive_loss = -(
            torch.log_softmax(positive_logits, dim=-1) * positives
        ).sum(dim=-1)

        negative_mask = 1.0 - positives
        negative_logits = selected_logits.masked_fill(negative_mask == 0, floor)
        negative_loss = -(
            torch.log_softmax(negative_logits, dim=-1) * threshold
        ).sum(dim=-1)

        return (positive_loss + negative_loss).mean()

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict every class scoring above the learned threshold.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.

        Returns:
            Boolean predictions of shape ``[B, P, R]`` with column zero cleared.
        """
        predictions = logits > logits[..., 0:1]
        predictions[..., 0] = False
        return predictions

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Score each class by how far it sits above the learned threshold.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.

        Returns:
            Confidences of shape ``[B, P, R]``, crossing ``0.5`` exactly where
            :meth:`decode` flips.
        """
        return torch.sigmoid(logits - logits[..., 0:1])


class BinaryRelationLoss(nn.Module):
    """Binary cross entropy objective with a fixed probability threshold.

    Kept as an alternative to :class:`AdaptiveThresholdLoss` so the decision rule
    can be swapped without touching the trainer. The threshold is tunable on the
    evaluation split by :class:`~nano_re.training.metrics.ThresholdSearch`.

    Args:
        threshold: Sigmoid probability above which a relation is predicted.
        positive_weight: Optional up-weighting of positive targets, which
            counteracts the sparsity of gold relations.
    """

    def __init__(
        self, threshold: float = 0.5, positive_weight: float | None = None
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self._positive_weight = positive_weight

    @property
    def name(self) -> str:
        """Identifier written into training reports and the model card."""
        return "bce"

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked binary cross entropy over the relation classes.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.
            targets: Multi-hot targets, shape ``[B, P, R]``.
            pair_mask: Validity flag per pair, shape ``[B, P]``.

        Returns:
            A scalar loss tensor averaged over unpadded pairs.
        """
        selected_logits, selected_targets = _select_valid_pairs(
            logits, targets, pair_mask
        )
        if selected_logits.numel() == 0:
            return logits.sum() * 0.0

        weight = None
        if self._positive_weight is not None:
            weight = torch.full(
                (selected_logits.shape[-1],),
                self._positive_weight,
                device=selected_logits.device,
                dtype=selected_logits.dtype,
            )
        return nn.functional.binary_cross_entropy_with_logits(
            selected_logits[:, 1:],
            selected_targets[:, 1:],
            pos_weight=None if weight is None else weight[1:],
        )

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict every class whose probability exceeds the threshold.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.

        Returns:
            Boolean predictions of shape ``[B, P, R]`` with column zero cleared.
        """
        predictions = torch.sigmoid(logits) > self.threshold
        predictions[..., 0] = False
        return predictions

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Score each class by its probability rescaled around the threshold.

        Rescaling keeps the confidence crossing ``0.5`` exactly where
        :meth:`decode` flips, whatever threshold was tuned.

        Args:
            logits: Relation scores, shape ``[B, P, R]``.

        Returns:
            Confidences of shape ``[B, P, R]``.
        """
        probabilities = torch.sigmoid(logits)
        below = 0.5 * probabilities / max(self.threshold, 1e-6)
        above = 0.5 + 0.5 * (probabilities - self.threshold) / max(
            1.0 - self.threshold, 1e-6
        )
        return torch.where(probabilities <= self.threshold, below, above)


class MultiTaskLoss(nn.Module):
    """Weighted combination of the token and relation objectives.

    Implements ``L_total = alpha * L_NER + beta * L_RE``.

    Args:
        relation_objective: Strategy providing the relation loss and decoder.
        ner_weight: Alpha coefficient applied to the token classification loss.
        relation_weight: Beta coefficient applied to the relation loss.
        ignore_index: Label value marking ignored token positions.
    """

    def __init__(
        self,
        relation_objective: RelationObjective,
        ner_weight: float = 1.0,
        relation_weight: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.relation_objective = relation_objective
        self.ner_weight = ner_weight
        self.relation_weight = relation_weight
        self._token_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        ner_logits: torch.Tensor,
        ner_labels: torch.Tensor,
        relation_logits: torch.Tensor,
        relation_labels: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> MultiTaskLossOutput:
        """Compute both task losses and their weighted sum.

        Args:
            ner_logits: Token scores, shape ``[B, S, L]``.
            ner_labels: Token targets, shape ``[B, S]``.
            relation_logits: Relation scores, shape ``[B, P, R]``.
            relation_labels: Relation targets, shape ``[B, P, R]``.
            pair_mask: Validity flag per pair, shape ``[B, P]``.

        Returns:
            The individual and combined loss terms.
        """
        ner_loss = self._token_loss(
            ner_logits.reshape(-1, ner_logits.shape[-1]), ner_labels.reshape(-1)
        )
        relation_loss = self.relation_objective.loss(
            relation_logits, relation_labels, pair_mask
        )
        total = self.ner_weight * ner_loss + self.relation_weight * relation_loss
        return MultiTaskLossOutput(total=total, ner=ner_loss, relation=relation_loss)


def build_relation_objective(
    name: str, threshold: float = 0.5
) -> RelationObjective:
    """Instantiate a relation objective by configuration name.

    Args:
        name: Either ``adaptive_threshold`` or ``bce``.
        threshold: Probability threshold used by the ``bce`` objective.

    Returns:
        The requested objective.

    Raises:
        ValueError: If the name is not a known objective.
    """
    if name == "adaptive_threshold":
        return AdaptiveThresholdLoss()
    if name == "bce":
        return BinaryRelationLoss(threshold=threshold)
    raise ValueError(
        f"Unknown relation objective {name!r}. Expected 'adaptive_threshold' or 'bce'."
    )


def _select_valid_pairs(
    logits: torch.Tensor, targets: torch.Tensor, pair_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batch and drop padded pair slots.

    Args:
        logits: Relation scores, shape ``[B, P, R]``.
        targets: Relation targets, shape ``[B, P, R]``.
        pair_mask: Validity flag per pair, shape ``[B, P]``.

    Returns:
        Flattened logits and targets containing only real pairs.
    """
    num_relations = logits.shape[-1]
    flat_logits = logits.reshape(-1, num_relations)
    flat_targets = targets.reshape(-1, num_relations)
    keep = pair_mask.reshape(-1) > 0
    return flat_logits[keep], flat_targets[keep]
