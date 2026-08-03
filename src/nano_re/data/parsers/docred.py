"""Parser for DocRED-shaped corpora, of which Re-DocRED is the one in use.

The format stores mention offsets relative to their sentence, so the parser
computes cumulative sentence offsets once and hands every downstream stage
document-global word indices.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...schema import SREDFM_TYPE_MAP, canonical_entity_type
from ..document import Document, Entity, Mention, RelationTriple


class DocRedParser:
    """Converts DocRED-shaped records into the internal representation.

    Args:
        inventory: Optional accumulator recording every relation encountered, so
            the label schema reflects the corpora actually used.
    """

    def __init__(self, inventory=None) -> None:
        self._inventory = inventory

    def parse(self, record: Mapping[str, Any], index: int) -> Document:
        """Convert a single raw record.

        Args:
            record: Raw object with ``sents``, ``vertexSet`` and ``labels``.
            index: Positional index used to build a fallback identifier.

        Returns:
            The parsed :class:`Document`.
        """
        sentences = record.get("sents") or []
        words: list[str] = []
        sentence_offsets: list[int] = []
        for sentence in sentences:
            sentence_offsets.append(len(words))
            words.extend(sentence)

        entities = self._parse_entities(record.get("vertexSet") or [], sentence_offsets)
        raw_labels = record.get("labels")
        relations = self._parse_relations(raw_labels or [], len(entities))

        return Document(
            doc_id=str(record.get("title") or f"document-{index}"),
            words=tuple(words),
            sentence_offsets=tuple(sentence_offsets),
            entities=entities,
            relations=relations,
            has_labels=raw_labels is not None,
            metadata={"language": str(record.get("lan") or "en")},
        )

    def parse_all(self, records) -> list[Document]:
        """Convert an iterable of raw records.

        Args:
            records: Iterable of raw DocRED-shaped objects.

        Returns:
            A list of parsed documents in input order.
        """
        return [self.parse(record, index) for index, record in enumerate(records)]

    def _parse_entities(
        self, vertex_set: list[list[Mapping[str, Any]]], sentence_offsets: list[int]
    ) -> tuple[Entity, ...]:
        """Convert coreference clusters into entities with global offsets.

        Args:
            vertex_set: The ``vertexSet`` payload.
            sentence_offsets: Word index at which each sentence starts.

        Returns:
            The entity clusters, preserving their original order.
        """
        entities: list[Entity] = []
        for cluster in vertex_set:
            mentions: list[Mention] = []
            raw_type = ""
            for raw_mention in cluster:
                sentence_id = int(raw_mention.get("sent_id", 0))
                if sentence_id >= len(sentence_offsets):
                    continue
                offset = sentence_offsets[sentence_id]
                start, end = raw_mention["pos"]
                raw_type = raw_type or str(raw_mention.get("type", "MISC"))
                mentions.append(
                    Mention(
                        text=str(raw_mention.get("name", "")),
                        start=offset + int(start),
                        end=offset + int(end),
                        sentence_id=sentence_id,
                    )
                )
            entities.append(
                Entity(
                    entity_type=canonical_entity_type(
                        raw_type or "MISC", SREDFM_TYPE_MAP
                    ),
                    mentions=tuple(mentions),
                )
            )
        return tuple(entities)

    def _parse_relations(
        self, raw_labels: list[Mapping[str, Any]], num_entities: int
    ) -> tuple[RelationTriple, ...]:
        """Convert gold labels into relation triples.

        Args:
            raw_labels: The ``labels`` payload.
            num_entities: Number of entities available for bounds checking.

        Returns:
            Relation triples whose endpoints exist in the document.
        """
        triples: list[RelationTriple] = []
        for label in raw_labels:
            head = int(label["h"])
            tail = int(label["t"])
            relation = str(label["r"])
            if head >= num_entities or tail >= num_entities or head == tail:
                continue
            if self._inventory is not None:
                self._inventory.add(relation)
            triples.append(
                RelationTriple(head=head, tail=tail, relation=relation)
            )
        return tuple(triples)
