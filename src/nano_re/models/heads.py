"""Task heads and the entity pooling layer.

Every operation here is shape polymorphic and free of data dependent control
flow, which is what allows the assembled model to export to ONNX with dynamic
batch, sequence, entity and pair axes.
"""

from __future__ import annotations

import torch
from torch import nn


class TokenClassificationHead(nn.Module):
    """Projects token representations onto BIO tag scores.

    Args:
        hidden_size: Width of the incoming token representations.
        num_labels: Number of BIO tags.
        dropout: Dropout probability applied before the projection.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score every token position.

        Args:
            hidden_states: Token representations, shape ``[B, S, H]``.

        Returns:
            BIO logits, shape ``[B, S, num_labels]``.
        """
        return self.classifier(self.dropout(hidden_states))


class EntityPooler(nn.Module):
    """Pools mention tokens into one vector per entity.

    Pooling is expressed as a single batched matrix product against the
    normalised mention mask. There is no gather over variable length mention
    lists, so the entity axis stays dynamic in the exported graph.
    """

    def forward(
        self, hidden_states: torch.Tensor, mention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute one representation per entity slot.

        Args:
            hidden_states: Token representations, shape ``[B, S, H]``.
            mention_mask: Row-normalised pooling weights, shape ``[B, E, S]``.

        Returns:
            Entity representations, shape ``[B, E, H]``.
        """
        return torch.bmm(mention_mask, hidden_states)


class PairwiseRelationHead(nn.Module):
    """Scores ordered entity pairs against the relation inventory.

    The pair feature concatenates the head and tail vectors with their element
    wise product and absolute difference. The product captures agreement and the
    difference captures direction, which a plain concatenation alone leaves the
    first layer to discover.

    Args:
        hidden_size: Width of an entity representation.
        num_relations: Number of relation classes including the ``NA`` slot.
        pair_hidden_size: Width of the hidden layer.
        dropout: Dropout probability applied to the pair feature and the hidden
            activation.
        use_context: Whether a pair context vector is supplied. When it is, each
            entity vector is projected together with that context before the
            pair feature is built, so the same entity is represented differently
            depending on which other entity it is being compared against.
    """

    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        pair_hidden_size: int = 512,
        dropout: float = 0.1,
        use_context: bool = False,
    ) -> None:
        super().__init__()
        self.use_context = use_context
        if use_context:
            self.head_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.tail_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, pair_hidden_size),
            nn.GELU(),
            nn.LayerNorm(pair_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_size, num_relations),
        )

    def forward(
        self,
        entity_representations: torch.Tensor,
        pair_index: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score every candidate pair.

        Args:
            entity_representations: Entity vectors, shape ``[B, E, H]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.
            context: Optional pair context vectors, shape ``[B, P, H]``.

        Returns:
            Relation logits, shape ``[B, P, num_relations]``.
        """
        head = self._gather(entity_representations, pair_index[:, :, 0])
        tail = self._gather(entity_representations, pair_index[:, :, 1])
        if self.use_context and context is not None:
            head = torch.tanh(self.head_projection(torch.cat([head, context], dim=-1)))
            tail = torch.tanh(self.tail_projection(torch.cat([tail, context], dim=-1)))
        features = torch.cat(
            [head, tail, head * tail, torch.abs(head - tail)], dim=-1
        )
        return self.mlp(features)

    @staticmethod
    def _gather(
        entity_representations: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Select entity vectors by index along the entity axis.

        Args:
            entity_representations: Entity vectors, shape ``[B, E, H]``.
            indices: Entity rows to select, shape ``[B, P]``.

        Returns:
            Selected vectors, shape ``[B, P, H]``.
        """
        hidden_size = entity_representations.shape[-1]
        expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        return torch.gather(entity_representations, 1, expanded)


class LocalizedContextPooler(nn.Module):
    """Builds a context vector specific to each candidate entity pair.

    Pooling an entity's own mentions says what the entity is; it says nothing
    about which part of the document relates this entity to that one. Localized
    context pooling reads that from the encoder's own attention: the tokens both
    entities attend to are the tokens that connect them.

    The construction follows Zhou et al., "Document-Level Relation Extraction
    with Adaptive Thresholding and Localized Context Pooling" (AAAI 2021). This
    is the second half of that method; the adaptive threshold loss is the first.

    Every step is a batched matrix product, so the entity and pair axes stay
    dynamic in the exported graph.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one context vector per candidate pair.

        Args:
            hidden_states: Token representations, shape ``[B, S, H]``.
            attention: Head-averaged attention, shape ``[B, S, S]``.
            mention_mask: Row-normalised entity pooling weights, ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            Pair context vectors, shape ``[B, P, H]``.
        """
        entity_attention = torch.bmm(mention_mask, attention)
        head = self._gather(entity_attention, pair_index[:, :, 0])
        tail = self._gather(entity_attention, pair_index[:, :, 1])
        overlap = head * tail
        weights = overlap / overlap.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.bmm(weights, hidden_states)

    @staticmethod
    def _gather(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Select rows by index along the entity axis.

        Args:
            source: Per-entity rows, shape ``[B, E, S]``.
            indices: Entity rows to select, shape ``[B, P]``.

        Returns:
            Selected rows, shape ``[B, P, S]``.
        """
        width = source.shape[-1]
        expanded = indices.unsqueeze(-1).expand(-1, -1, width)
        return torch.gather(source, 1, expanded)
