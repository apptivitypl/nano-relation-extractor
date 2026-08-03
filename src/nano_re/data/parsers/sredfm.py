"""Parser for the SREDFM and REDFM relation corpora.

These corpora annotate entities by character offset into raw text, while the
model works in words. The parser therefore tokenises the text once, records
where every word starts and ends, and translates each annotation onto the word
range it overlaps.

Overlap rather than exact match is deliberate: an annotation whose boundary
falls mid-word still identifies the word it lands in, and silently dropping such
mentions would quietly shrink the training signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...schema import SREDFM_TYPE_MAP, canonical_entity_type
from ..document import Document, Entity, Mention, RelationTriple

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class WordSpan:
    """One word with its character range.

    Attributes:
        text: Surface form of the word.
        start: Inclusive character offset.
        end: Exclusive character offset.
    """

    text: str
    start: int
    end: int


class SredfmParser:
    """Converts SREDFM records into the internal document representation.

    Args:
        inventory: Optional accumulator recording every relation encountered, so
            the label schema can be built from the corpus rather than assumed.
    """

    def __init__(self, inventory=None) -> None:
        self._inventory = inventory

    def parse(self, record: dict, index: int) -> Document:
        """Convert a single raw record.

        Args:
            record: Raw SREDFM object with ``text``, ``entities`` and
                ``relations``.
            index: Positional index used to build a fallback identifier.

        Returns:
            The parsed :class:`Document`. Documents whose annotations cannot be
            aligned produce an empty entity list rather than raising.
        """
        text = record.get("text") or ""
        words = self._tokenise(text)
        entities, key_to_index = self._parse_entities(record, words)
        relations = self._parse_relations(record, key_to_index)

        return Document(
            doc_id=str(record.get("docid") or record.get("uri") or f"document-{index}"),
            words=tuple(word.text for word in words),
            sentence_offsets=(0,),
            entities=entities,
            relations=relations,
            has_labels=True,
            metadata={"language": str(record.get("lan") or "")},
        )

    def parse_all(self, records) -> list[Document]:
        """Convert an iterable of raw records.

        Args:
            records: Iterable of raw SREDFM objects.

        Returns:
            A list of parsed documents in input order.
        """
        return [self.parse(record, index) for index, record in enumerate(records)]

    def _tokenise(self, text: str) -> list[WordSpan]:
        """Split text into words that remember their character positions.

        Args:
            text: Raw document text.

        Returns:
            The word spans, in order.
        """
        return [
            WordSpan(text=match.group(), start=match.start(), end=match.end())
            for match in TOKEN_PATTERN.finditer(text)
        ]

    def _locate(
        self, words: list[WordSpan], boundaries
    ) -> tuple[int, int] | None:
        """Translate a character range onto a word range.

        Args:
            words: Word spans of the document.
            boundaries: Two element character range from the annotation.

        Returns:
            The half-open word range, or ``None`` when nothing overlaps.
        """
        if not boundaries or len(boundaries) != 2:
            return None
        start, end = int(boundaries[0]), int(boundaries[1])
        if end <= start:
            return None
        covered = [
            position
            for position, word in enumerate(words)
            if word.start < end and word.end > start
        ]
        if not covered:
            return None
        return covered[0], covered[-1] + 1

    def _entity_key(self, payload: dict) -> str:
        """Build the identity under which mentions are grouped.

        The corpus supplies a Wikidata identifier for most entities, which
        groups mentions far more reliably than surface form: an inflected or
        abbreviated second mention still carries the same identifier. Surface
        form is the fallback for the roughly one entity in three without one.

        Args:
            payload: Raw entity or relation argument.

        Returns:
            A grouping key.
        """
        uri = str(payload.get("uri") or "").strip()
        if uri:
            return uri
        return "surface:" + str(payload.get("surfaceform") or "").strip().lower()

    def _parse_entities(
        self, record: dict, words: list[WordSpan]
    ) -> tuple[tuple[Entity, ...], dict[str, int]]:
        """Group annotated mentions into entity clusters.

        Args:
            record: Raw SREDFM object.
            words: Word spans of the document.

        Returns:
            The entity clusters and a mapping from grouping key to cluster index.
        """
        grouped: dict[str, list[Mention]] = {}
        types: dict[str, str] = {}
        order: list[str] = []

        arguments = list(record.get("entities") or [])
        for relation in record.get("relations") or []:
            arguments.extend(
                [relation.get("subject") or {}, relation.get("object") or {}]
            )

        for payload in arguments:
            span = self._locate(words, payload.get("boundaries"))
            if span is None:
                continue
            key = self._entity_key(payload)
            if key not in grouped:
                grouped[key] = []
                types[key] = canonical_entity_type(
                    str(payload.get("type") or "MISC"), SREDFM_TYPE_MAP
                )
                order.append(key)
            mention = Mention(
                text=str(payload.get("surfaceform") or ""),
                start=span[0],
                end=span[1],
                sentence_id=0,
            )
            if mention not in grouped[key]:
                grouped[key].append(mention)

        entities = tuple(
            Entity(entity_type=types[key], mentions=tuple(grouped[key]))
            for key in order
        )
        return entities, {key: index for index, key in enumerate(order)}

    def _parse_relations(
        self, record: dict, key_to_index: dict[str, int]
    ) -> tuple[RelationTriple, ...]:
        """Convert annotated relations into triples over entity indices.

        Args:
            record: Raw SREDFM object.
            key_to_index: Mapping from grouping key to cluster index.

        Returns:
            The relation triples whose endpoints both resolved.
        """
        triples: list[RelationTriple] = []
        for relation in record.get("relations") or []:
            head = key_to_index.get(self._entity_key(relation.get("subject") or {}))
            tail = key_to_index.get(self._entity_key(relation.get("object") or {}))
            predicate = relation.get("predicate") or {}
            relation_id = str(predicate.get("uri") or "").strip()
            if head is None or tail is None or head == tail or not relation_id:
                continue
            if self._inventory is not None:
                self._inventory.add(
                    relation_id, str(predicate.get("surfaceform") or "") or None
                )
            triples.append(
                RelationTriple(head=head, tail=tail, relation=relation_id)
            )
        return tuple(triples)
