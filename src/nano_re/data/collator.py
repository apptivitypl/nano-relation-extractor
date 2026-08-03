"""Batching of encoded documents.

Documents differ along four axes at once: sequence length, entity count, pair
count and relation cardinality. The collator pads the first three and emits
explicit masks so padded slots contribute to neither pooling, loss nor metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .encoder import EncodedDocument


@dataclass
class MultiTaskBatch:
    """A padded batch of documents ready for the model.

    Attributes:
        input_ids: Sub-word identifiers, shape ``[B, S]``.
        attention_mask: Padding mask over sub-words, shape ``[B, S]``.
        ner_labels: BIO targets with ``-100`` on ignored positions, ``[B, S]``.
        mention_mask: Entity pooling weights, shape ``[B, E, S]``.
        entity_mask: Validity flag per entity slot, shape ``[B, E]``.
        pair_index: Head and tail entity rows per pair, shape ``[B, P, 2]``.
        pair_mask: Validity flag per pair slot, shape ``[B, P]``. A document from
            a corpus that annotates entities but not relations carries zeros
            across the whole row. Because both the relation loss and the
            relation metric already select on this mask, that single assignment
            is what keeps an entity-only corpus from teaching the relation head
            that every pair is ``NA``.
        relation_labels: Multi-hot relation targets, shape ``[B, P, R]``.
        doc_ids: Source document identifier per batch element.
        source_entity_ids: Original entity indices per batch element, used to
            reconstruct corpus-level triples during evaluation.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    ner_labels: torch.Tensor
    mention_mask: torch.Tensor
    entity_mask: torch.Tensor
    pair_index: torch.Tensor
    pair_mask: torch.Tensor
    relation_labels: torch.Tensor
    doc_ids: tuple[str, ...]
    source_entity_ids: tuple[tuple[int, ...], ...]

    def to(self, device: torch.device) -> "MultiTaskBatch":
        """Move every tensor in the batch to a device.

        Args:
            device: Target device.

        Returns:
            A new batch whose tensors live on ``device``.
        """
        return MultiTaskBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            ner_labels=self.ner_labels.to(device),
            mention_mask=self.mention_mask.to(device),
            entity_mask=self.entity_mask.to(device),
            pair_index=self.pair_index.to(device),
            pair_mask=self.pair_mask.to(device),
            relation_labels=self.relation_labels.to(device),
            doc_ids=self.doc_ids,
            source_entity_ids=self.source_entity_ids,
        )

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Return only the tensors accepted by the model's forward signature.

        Returns:
            Keyword arguments for :meth:`NanoREModel.forward`.
        """
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "mention_mask": self.mention_mask,
            "pair_index": self.pair_index,
        }


class MultiTaskCollator:
    """Pads encoded documents into a :class:`MultiTaskBatch`.

    Args:
        pad_token_id: Identifier used to pad ``input_ids``.
        label_pad_id: Value marking ignored NER positions.
    """

    def __init__(self, pad_token_id: int, label_pad_id: int = -100) -> None:
        self._pad_token_id = pad_token_id
        self._label_pad_id = label_pad_id

    def __call__(
        self, documents: list[EncodedDocument | None]
    ) -> MultiTaskBatch | None:
        """Collate a list of encoded documents.

        Lazy encoding yields ``None`` for documents that cannot produce a
        relation. They are dropped here so no caller has to pre-filter, and a
        batch that loses every member becomes ``None`` rather than an empty
        tensor set that downstream code would have to special-case anyway.

        Args:
            documents: Encoded documents belonging to one batch, possibly
                containing ``None`` placeholders.

        Returns:
            The padded batch, or ``None`` when nothing usable remains.
        """
        documents = [item for item in documents if item is not None]
        if not documents:
            return None
        batch_size = len(documents)
        max_sequence = max(int(item.input_ids.shape[0]) for item in documents)
        max_entities = max(item.num_entities for item in documents)
        max_pairs = max(item.num_pairs for item in documents)
        num_relations = int(documents[0].relation_labels.shape[-1])

        input_ids = torch.full(
            (batch_size, max_sequence), self._pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_sequence), dtype=torch.long)
        ner_labels = torch.full(
            (batch_size, max_sequence), self._label_pad_id, dtype=torch.long
        )
        mention_mask = torch.zeros(
            (batch_size, max_entities, max_sequence), dtype=torch.float32
        )
        entity_mask = torch.zeros((batch_size, max_entities), dtype=torch.float32)
        pair_index = torch.zeros((batch_size, max_pairs, 2), dtype=torch.long)
        pair_mask = torch.zeros((batch_size, max_pairs), dtype=torch.float32)
        relation_labels = torch.zeros(
            (batch_size, max_pairs, num_relations), dtype=torch.float32
        )

        for row, item in enumerate(documents):
            sequence = int(item.input_ids.shape[0])
            entities = item.num_entities
            pairs = item.num_pairs

            input_ids[row, :sequence] = item.input_ids
            attention_mask[row, :sequence] = item.attention_mask
            ner_labels[row, :sequence] = item.ner_labels
            mention_mask[row, :entities, :sequence] = item.mention_mask
            entity_mask[row, :entities] = 1.0
            pair_index[row, :pairs] = item.pair_index
            relation_labels[row, :pairs] = item.relation_labels
            if item.has_relation_supervision:
                pair_mask[row, :pairs] = 1.0

        return MultiTaskBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ner_labels=ner_labels,
            mention_mask=mention_mask,
            entity_mask=entity_mask,
            pair_index=pair_index,
            pair_mask=pair_mask,
            relation_labels=relation_labels,
            doc_ids=tuple(item.doc_id for item in documents),
            source_entity_ids=tuple(item.source_entity_ids for item in documents),
        )
