"""Translation of raw DocRED records into :class:`Document` objects.

DocRED stores mention offsets relative to their sentence. Every consumer needs
document-global word indices, so the conversion happens exactly once, here.
"""

from __future__ import annotations

from typing import Any, Mapping

from .document import Document, Entity, Mention, RelationTriple


class DocRedParser:
    """Converts raw DocRED records into the internal document representation."""

    def parse(self, record: Mapping[str, Any], index: int) -> Document:
        """Convert a single raw record.

        Args:
            record: Raw DocRED object with ``sents``, ``vertexSet`` and,
                for annotated splits, ``labels``.
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
        has_labels = raw_labels is not None
        relations = self._parse_relations(raw_labels or [], len(entities))

        return Document(
            doc_id=str(record.get("title") or f"document-{index}"),
            words=tuple(words),
            sentence_offsets=tuple(sentence_offsets),
            entities=entities,
            relations=relations,
            has_labels=has_labels,
        )

    def parse_all(self, records) -> list[Document]:
        """Convert an iterable of raw records.

        Args:
            records: Iterable of raw DocRED objects.

        Returns:
            A list of parsed documents in input order.
        """
        return [self.parse(record, index) for index, record in enumerate(records)]

    def _parse_entities(
        self, vertex_set: list[list[Mapping[str, Any]]], sentence_offsets: list[int]
    ) -> tuple[Entity, ...]:
        """Convert coreference clusters into entities with global offsets.

        Args:
            vertex_set: DocRED ``vertexSet`` payload.
            sentence_offsets: Word index at which each sentence starts.

        Returns:
            The parsed entity clusters, preserving their original order.
        """
        entities: list[Entity] = []
        for cluster in vertex_set:
            mentions: list[Mention] = []
            entity_type = ""
            for raw_mention in cluster:
                sentence_id = int(raw_mention.get("sent_id", 0))
                if sentence_id >= len(sentence_offsets):
                    continue
                offset = sentence_offsets[sentence_id]
                start, end = raw_mention["pos"]
                entity_type = entity_type or str(raw_mention.get("type", "MISC"))
                mentions.append(
                    Mention(
                        text=str(raw_mention.get("name", "")),
                        start=offset + int(start),
                        end=offset + int(end),
                        sentence_id=sentence_id,
                    )
                )
            entities.append(
                Entity(entity_type=entity_type or "MISC", mentions=tuple(mentions))
            )
        return tuple(entities)

    def _parse_relations(
        self, raw_labels: list[Mapping[str, Any]], num_entities: int
    ) -> tuple[RelationTriple, ...]:
        """Convert gold labels into relation triples.

        Args:
            raw_labels: DocRED ``labels`` payload.
            num_entities: Number of entities available for bounds checking.

        Returns:
            Relation triples whose endpoints exist in the document.
        """
        triples: list[RelationTriple] = []
        for label in raw_labels:
            head = int(label["h"])
            tail = int(label["t"])
            if head >= num_entities or tail >= num_entities or head == tail:
                continue
            triples.append(
                RelationTriple(head=head, tail=tail, relation=str(label["r"]))
            )
        return tuple(triples)
