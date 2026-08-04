"""Evaluation metrics for both tasks.

Both metrics are accumulators: the caller streams batches in and reads a single
result at the end. Nothing here knows about models, devices or ONNX, so the same
objects score PyTorch and ONNX Runtime predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from ..schema import LabelSchema


@dataclass(frozen=True)
class ClassificationScore:
    """Precision, recall and F1 for one task.

    Attributes:
        precision: Micro-averaged precision.
        recall: Micro-averaged recall.
        f1: Micro-averaged F1.
        support: Number of gold items the score was computed over.
    """

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON compatible representation of the score."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ClassificationScore":
        """Rebuild a score from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed score.
        """
        return cls(
            precision=float(payload["precision"]),
            recall=float(payload["recall"]),
            f1=float(payload["f1"]),
            support=int(payload["support"]),
        )


def decode_bio_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """Extract typed spans from a BIO tag sequence.

    A span ends when a tag is ``O``, when a new ``B-`` begins, or when an ``I-``
    changes type. An ``I-`` with no preceding ``B-`` opens a span rather than
    being discarded, which is the forgiving reading and matches how the decoder
    used at inference behaves.

    Args:
        tags: BIO tag per token.

    Returns:
        Spans as start, end and type. A set, because span level scoring counts
        each distinct span once regardless of order.
    """
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    active = ""

    for position, tag in enumerate(tags):
        prefix, _, entity_type = tag.partition("-")
        if prefix == "B" or (prefix == "I" and entity_type != active):
            if start is not None:
                spans.add((start, position, active))
            start = position
            active = entity_type
        elif prefix not in {"B", "I"}:
            if start is not None:
                spans.add((start, position, active))
            start = None
            active = ""

    if start is not None:
        spans.add((start, len(tags), active))
    return spans


class NerMetric:
    """Span level NER scorer.

    Scoring is span level rather than token level: a mention is correct only
    when its boundaries and type both match, which is what a consumer building
    nodes from the output actually needs.

    The arithmetic is implemented here rather than delegated. The usual library
    for it, seqeval, was last released in 2020 and ships no wheel, so installing
    it runs a setup script that fails on current Python. That is a poor trade for
    counting set intersections.

    Args:
        schema: Label vocabularies used to decode indices into BIO tags.
    """

    def __init__(self, schema: LabelSchema) -> None:
        self._schema = schema
        self._id_to_bio = schema.id_to_bio
        self._true_positive = 0
        self._predicted = 0
        self._gold = 0

    def reset(self) -> None:
        """Discard everything accumulated so far."""
        self._true_positive = 0
        self._predicted = 0
        self._gold = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate one batch of token predictions.

        Args:
            logits: Token scores, shape ``[B, S, L]``.
            labels: Token targets with ``-100`` on ignored positions, ``[B, S]``.
        """
        predicted_ids = logits.argmax(dim=-1).detach().cpu().numpy()
        label_ids = labels.detach().cpu().numpy()
        for row_predictions, row_labels in zip(predicted_ids, label_ids):
            kept = row_labels != -100
            predicted = decode_bio_spans(
                [self._id_to_bio[int(value)] for value in row_predictions[kept]]
            )
            gold = decode_bio_spans(
                [self._id_to_bio[int(value)] for value in row_labels[kept]]
            )
            self._true_positive += len(predicted & gold)
            self._predicted += len(predicted)
            self._gold += len(gold)

    def compute(self) -> ClassificationScore:
        """Return the accumulated span level score.

        Returns:
            Micro-averaged precision, recall and F1 over entity spans.
        """
        precision = (
            self._true_positive / self._predicted if self._predicted else 0.0
        )
        recall = self._true_positive / self._gold if self._gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return ClassificationScore(
            precision=precision,
            recall=recall,
            f1=f1,
            support=self._gold,
        )


class RelationMetric:
    """Micro-averaged relation scorer over candidate entity pairs.

    Scores are computed over every pair the model was actually asked about. Gold
    triples lost to sequence truncation never reach the model, so they are
    tracked separately and reported as a recall ceiling rather than being
    silently excluded from recall.
    """

    def __init__(self) -> None:
        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._unreachable_gold = 0

    def reset(self) -> None:
        """Discard everything accumulated so far."""
        self._predictions.clear()
        self._targets.clear()
        self._unreachable_gold = 0

    def add_unreachable_gold(self, count: int) -> None:
        """Record gold triples that truncation removed from the candidate set.

        Args:
            count: Number of unreachable gold triples.
        """
        self._unreachable_gold += count

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> None:
        """Accumulate one batch of relation predictions.

        Args:
            predictions: Boolean predictions, shape ``[B, P, R]``.
            targets: Multi-hot targets, shape ``[B, P, R]``.
            pair_mask: Validity flag per pair, shape ``[B, P]``.
        """
        keep = pair_mask.reshape(-1).detach().cpu().numpy() > 0
        num_relations = predictions.shape[-1]
        flat_predictions = (
            predictions.reshape(-1, num_relations).detach().cpu().numpy()[keep]
        )
        flat_targets = targets.reshape(-1, num_relations).detach().cpu().numpy()[keep]
        self._predictions.append(flat_predictions[:, 1:].astype(np.int8))
        self._targets.append(flat_targets[:, 1:].astype(np.int8))

    def compute(self) -> ClassificationScore:
        """Return the accumulated micro-averaged relation score.

        Returns:
            Micro-averaged precision, recall and F1 over relation triples.
        """
        if not self._targets:
            return ClassificationScore(0.0, 0.0, 0.0, 0)
        y_pred = np.concatenate(self._predictions, axis=0)
        y_true = np.concatenate(self._targets, axis=0)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        return ClassificationScore(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            support=int(y_true.sum()),
        )

    @property
    def recall_ceiling(self) -> float:
        """Highest recall achievable given gold triples lost to truncation."""
        reachable = sum(int(target.sum()) for target in self._targets)
        total = reachable + self._unreachable_gold
        return reachable / total if total else 1.0


@dataclass(frozen=True)
class EvaluationResult:
    """Combined evaluation outcome for both tasks.

    Attributes:
        ner: Span level NER score.
        relation: Micro-averaged relation score.
        relation_recall_ceiling: Highest relation recall reachable after
            truncation.
        loss: Mean total loss over the evaluation split, when available.
    """

    ner: ClassificationScore
    relation: ClassificationScore
    relation_recall_ceiling: float
    loss: float | None = None

    @property
    def combined_f1(self) -> float:
        """Mean of both task F1 scores, used for checkpoint selection."""
        return (self.ner.f1 + self.relation.f1) / 2.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the result."""
        return {
            "ner": self.ner.to_dict(),
            "relation": self.relation.to_dict(),
            "relation_recall_ceiling": self.relation_recall_ceiling,
            "combined_f1": self.combined_f1,
            "loss": self.loss,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EvaluationResult":
        """Rebuild an evaluation result from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed result.
        """
        loss = payload.get("loss")
        return cls(
            ner=ClassificationScore.from_dict(payload["ner"]),
            relation=ClassificationScore.from_dict(payload["relation"]),
            relation_recall_ceiling=float(payload["relation_recall_ceiling"]),
            loss=None if loss is None else float(loss),
        )


class ThresholdSearch:
    """Selects the probability threshold maximising relation F1.

    Only meaningful for the binary cross entropy objective. Adaptive thresholding
    learns its own per-pair threshold and needs no search.

    Args:
        candidates: Thresholds to evaluate.
    """

    def __init__(self, candidates: tuple[float, ...] | None = None) -> None:
        self._candidates = candidates or tuple(
            round(0.05 * step, 2) for step in range(1, 20)
        )

    def search(
        self, logits: torch.Tensor, targets: torch.Tensor, pair_mask: torch.Tensor
    ) -> tuple[float, float]:
        """Find the threshold with the best micro F1.

        Args:
            logits: Relation scores for the whole split, shape ``[N, R]``.
            targets: Multi-hot targets for the whole split, shape ``[N, R]``.
            pair_mask: Validity flag per pair, shape ``[N]``.

        Returns:
            The best threshold and the F1 it achieves.
        """
        keep = pair_mask.reshape(-1) > 0
        probabilities = torch.sigmoid(logits.reshape(-1, logits.shape[-1])[keep])
        y_true = targets.reshape(-1, targets.shape[-1])[keep][:, 1:].cpu().numpy()
        best_threshold = self._candidates[0]
        best_f1 = -1.0
        for threshold in self._candidates:
            y_pred = (probabilities[:, 1:] > threshold).cpu().numpy()
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = float(f1)
                best_threshold = threshold
        return best_threshold, best_f1
