"""Structured model outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultiTaskOutput:
    """Logits produced by both task heads for one batch.

    Attributes:
        ner_logits: Token classification scores, shape ``[B, S, num_bio_labels]``.
        relation_logits: Pair classification scores, shape
            ``[B, P, num_relation_labels]``.
        entity_representations: Pooled entity vectors, shape ``[B, E, H]``.
            Retained for inspection and probing, not required downstream.
    """

    ner_logits: torch.Tensor
    relation_logits: torch.Tensor
    entity_representations: torch.Tensor
