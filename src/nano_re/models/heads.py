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
    """

    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        pair_hidden_size: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, pair_hidden_size),
            nn.GELU(),
            nn.LayerNorm(pair_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_size, num_relations),
        )

    def forward(
        self, entity_representations: torch.Tensor, pair_index: torch.Tensor
    ) -> torch.Tensor:
        """Score every candidate pair.

        Args:
            entity_representations: Entity vectors, shape ``[B, E, H]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            Relation logits, shape ``[B, P, num_relations]``.
        """
        head = self._gather(entity_representations, pair_index[:, :, 0])
        tail = self._gather(entity_representations, pair_index[:, :, 1])
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
