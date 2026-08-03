"""Conversion of documents into model ready tensors.

The encoder owns every decision that couples the corpus to the architecture:
sub-word alignment, BIO tagging, mention pooling weights, candidate pair
construction and negative sampling. Keeping all of it here means the model never
sees raw text and the corpus never sees a tokenizer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from ..schema import LabelSchema
from .document import Document


@dataclass(frozen=True)
class EncodedDocument:
    """Tensors describing one document for the multi-task model.

    Attributes:
        doc_id: Identifier of the source document.
        input_ids: Sub-word identifiers, shape ``[S]``.
        attention_mask: Padding mask, shape ``[S]``.
        ner_labels: BIO label per sub-word with ``-100`` on ignored positions,
            shape ``[S]``.
        mention_mask: Row-normalised pooling weights per entity, shape ``[E, S]``.
        pair_index: Head and tail entity indices per candidate pair, shape
            ``[P, 2]``.
        relation_labels: Multi-hot relation targets, shape ``[P, R]``.
        entity_types: Gold entity type of each retained entity.
        source_entity_ids: Original entity index of each retained entity, used to
            map predictions back onto gold triples.
        dropped_relations: Gold triples lost because an endpoint was truncated
            away. Reported so the recall ceiling stays visible.
        has_relation_supervision: Whether the source corpus annotates relations.
            A corpus that only annotates entities must not teach the relation
            head that every pair is ``NA``, so this flag travels with the tensors
            into the collator and from there into the loss.
        language: Language of the source document, used in reporting.
    """

    doc_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    ner_labels: torch.Tensor
    mention_mask: torch.Tensor
    pair_index: torch.Tensor
    relation_labels: torch.Tensor
    entity_types: tuple[str, ...]
    source_entity_ids: tuple[int, ...]
    dropped_relations: int = 0
    has_relation_supervision: bool = True
    language: str = ""

    @property
    def num_entities(self) -> int:
        """Number of entities retained after truncation."""
        return int(self.mention_mask.shape[0])

    @property
    def num_pairs(self) -> int:
        """Number of candidate entity pairs."""
        return int(self.pair_index.shape[0])


class DocumentEncoder:
    """Turns :class:`Document` objects into :class:`EncodedDocument` tensors.

    Args:
        tokenizer: Fast tokenizer providing word to sub-word alignment.
        schema: Label vocabularies for both tasks.
        max_sequence_length: Maximum number of sub-word tokens per document.
        max_negative_pairs: Negative pairs sampled per document when
            ``sample_negatives`` is requested. Ignored during evaluation.
        seed: Seed for the negative sampling generator.

    Raises:
        ValueError: If the tokenizer does not support word alignment.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        schema: LabelSchema,
        max_sequence_length: int = 512,
        max_negative_pairs: int = 96,
        seed: int = 42,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "DocumentEncoder requires a fast tokenizer for word alignment."
            )
        self._tokenizer = tokenizer
        self._schema = schema
        self._max_sequence_length = max_sequence_length
        self._max_negative_pairs = max_negative_pairs
        self._rng = random.Random(seed)
        self._bio_to_id = schema.bio_to_id
        self._relation_to_id = schema.relation_to_id

    @property
    def schema(self) -> LabelSchema:
        """Label vocabularies used to produce targets."""
        return self._schema

    def encode(
        self, document: Document, sample_negatives: bool = False
    ) -> EncodedDocument | None:
        """Encode a single document.

        Args:
            document: Parsed source document.
            sample_negatives: When ``True`` only a sample of negative pairs is
                emitted, which bounds training memory. Evaluation must leave this
                ``False`` so every candidate pair is scored.

        Returns:
            The encoded document, or ``None`` when fewer than two entities
            survive truncation and no relation could ever be predicted.
        """
        if not document.words or not document.entities:
            return None

        encoding = self._tokenizer(
            list(document.words),
            is_split_into_words=True,
            truncation=True,
            max_length=self._max_sequence_length,
            return_tensors=None,
        )
        word_ids = encoding.word_ids(0) if hasattr(encoding, "word_ids") else None
        if word_ids is None:
            return None

        word_to_subwords = self._build_word_alignment(word_ids)
        sequence_length = len(encoding["input_ids"])

        kept_entities, source_entity_ids = self._select_entities(
            document, word_to_subwords
        )
        supervises_relations = document.has_labels
        if supervises_relations and len(kept_entities) < 2:
            return None
        if not kept_entities:
            return None

        mention_mask = self._build_mention_mask(
            document, kept_entities, word_to_subwords, sequence_length
        )
        ner_labels = self._build_ner_labels(document, word_to_subwords, sequence_length)
        pair_index, relation_labels, dropped = self._build_pairs(
            document, source_entity_ids, sample_negatives
        )

        return EncodedDocument(
            doc_id=document.doc_id,
            input_ids=torch.tensor(encoding["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(encoding["attention_mask"], dtype=torch.long),
            ner_labels=ner_labels,
            mention_mask=mention_mask,
            pair_index=pair_index,
            relation_labels=relation_labels,
            entity_types=tuple(
                document.entities[index].entity_type for index in source_entity_ids
            ),
            source_entity_ids=tuple(source_entity_ids),
            dropped_relations=dropped,
            has_relation_supervision=supervises_relations,
            language=document.metadata.get("language", ""),
        )

    def encode_all(
        self, documents, sample_negatives: bool = False
    ) -> list[EncodedDocument]:
        """Encode an iterable of documents, discarding unusable ones.

        Args:
            documents: Iterable of parsed documents.
            sample_negatives: Forwarded to :meth:`encode`.

        Returns:
            Every successfully encoded document, in input order.
        """
        encoded: list[EncodedDocument] = []
        for document in documents:
            item = self.encode(document, sample_negatives=sample_negatives)
            if item is not None:
                encoded.append(item)
        return encoded

    def _build_word_alignment(self, word_ids: list[int | None]) -> dict[int, list[int]]:
        """Group sub-word positions by their source word index.

        Args:
            word_ids: Per-position word index, ``None`` for special tokens.

        Returns:
            Mapping from word index to the sub-word positions covering it.
        """
        alignment: dict[int, list[int]] = {}
        for position, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            alignment.setdefault(word_id, []).append(position)
        return alignment

    def _select_entities(
        self, document: Document, word_to_subwords: dict[int, list[int]]
    ) -> tuple[list[int], list[int]]:
        """Determine which entities survive sequence truncation.

        Args:
            document: Source document.
            word_to_subwords: Word to sub-word alignment.

        Returns:
            A pair of identical index lists: the retained entity positions in the
            original ``vertexSet`` order.
        """
        retained = [
            index
            for index, entity in enumerate(document.entities)
            if any(
                word in word_to_subwords
                for mention in entity.mentions
                for word in range(mention.start, mention.end)
            )
        ]
        return retained, retained

    def _build_mention_mask(
        self,
        document: Document,
        kept_entities: list[int],
        word_to_subwords: dict[int, list[int]],
        sequence_length: int,
    ) -> torch.Tensor:
        """Build row-normalised pooling weights for each retained entity.

        Args:
            document: Source document.
            kept_entities: Retained entity indices.
            word_to_subwords: Word to sub-word alignment.
            sequence_length: Number of sub-word positions.

        Returns:
            A ``[E, S]`` float tensor whose rows sum to one.
        """
        mask = torch.zeros(len(kept_entities), sequence_length, dtype=torch.float32)
        for row, entity_index in enumerate(kept_entities):
            entity = document.entities[entity_index]
            for mention in entity.mentions:
                for word in range(mention.start, mention.end):
                    for position in word_to_subwords.get(word, ()):
                        mask[row, position] = 1.0
        totals = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return mask / totals

    def _build_ner_labels(
        self,
        document: Document,
        word_to_subwords: dict[int, list[int]],
        sequence_length: int,
    ) -> torch.Tensor:
        """Build BIO targets aligned to the first sub-word of each word.

        Args:
            document: Source document.
            word_to_subwords: Word to sub-word alignment.
            sequence_length: Number of sub-word positions.

        Returns:
            A ``[S]`` long tensor using ``-100`` for ignored positions.
        """
        labels = torch.full((sequence_length,), -100, dtype=torch.long)
        outside_id = self._bio_to_id["O"]
        for positions in word_to_subwords.values():
            labels[positions[0]] = outside_id

        for entity in document.entities:
            entity_type = entity.entity_type
            begin_id = self._bio_to_id.get(f"B-{entity_type}")
            inside_id = self._bio_to_id.get(f"I-{entity_type}")
            if begin_id is None or inside_id is None:
                continue
            for mention in entity.mentions:
                for offset, word in enumerate(range(mention.start, mention.end)):
                    positions = word_to_subwords.get(word)
                    if not positions:
                        continue
                    labels[positions[0]] = begin_id if offset == 0 else inside_id
        return labels

    def _build_pairs(
        self,
        document: Document,
        source_entity_ids: list[int],
        sample_negatives: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build candidate pairs and their multi-hot relation targets.

        Args:
            document: Source document.
            source_entity_ids: Retained entity indices in row order.
            sample_negatives: Whether to subsample negative pairs.

        Returns:
            The pair index tensor, the relation target tensor, and the number of
            gold triples lost to truncation.
        """
        remap = {source: row for row, source in enumerate(source_entity_ids)}
        positives: dict[tuple[int, int], set[int]] = {}
        dropped = 0
        for triple in document.relations:
            head = remap.get(triple.head)
            tail = remap.get(triple.tail)
            relation_id = self._relation_to_id.get(triple.relation)
            if head is None or tail is None or relation_id is None:
                dropped += 1
                continue
            positives.setdefault((head, tail), set()).add(relation_id)

        num_entities = len(source_entity_ids)
        all_pairs = [
            (head, tail)
            for head in range(num_entities)
            for tail in range(num_entities)
            if head != tail
        ]

        if sample_negatives:
            negatives = [pair for pair in all_pairs if pair not in positives]
            if len(negatives) > self._max_negative_pairs:
                negatives = self._rng.sample(negatives, self._max_negative_pairs)
            selected = list(positives) + negatives
            self._rng.shuffle(selected)
        else:
            selected = all_pairs

        if not selected:
            selected = [(0, 0)]

        pair_index = torch.tensor(selected, dtype=torch.long)
        relation_labels = torch.zeros(
            len(selected), self._schema.num_relation_labels, dtype=torch.float32
        )
        for row, pair in enumerate(selected):
            relation_ids = positives.get(pair)
            if relation_ids:
                for relation_id in relation_ids:
                    relation_labels[row, relation_id] = 1.0
            else:
                relation_labels[row, 0] = 1.0
        return pair_index, relation_labels, dropped
