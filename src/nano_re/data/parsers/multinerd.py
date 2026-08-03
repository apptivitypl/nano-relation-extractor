"""Parser for the MultiNERD entity recognition corpus.

MultiNERD arrives already tokenised and BIO tagged, so the work here is folding
its fifteen types onto the canonical inventory and reconstructing spans from
tags. Records carry no relations, which is why the resulting documents declare
``has_labels=False``: they supervise the token head only.
"""

from __future__ import annotations

from ...schema import MULTINERD_TAGS, MULTINERD_TYPE_MAP, canonical_entity_type
from ..document import Document, Entity, Mention


class MultiNerdParser:
    """Converts MultiNERD records into the internal document representation."""

    def parse(self, record: dict, index: int) -> Document:
        """Convert a single raw record.

        Args:
            record: Raw object with ``tokens`` and ``ner_tags``.
            index: Positional index used to build a fallback identifier.

        Returns:
            The parsed :class:`Document`, carrying entities but no relations.
        """
        words = [str(token) for token in record.get("tokens") or []]
        tags = [int(tag) for tag in record.get("ner_tags") or []]
        entities = self._parse_entities(words, tags)

        return Document(
            doc_id=f"multinerd-{record.get('lang', 'xx')}-{index}",
            words=tuple(words),
            sentence_offsets=(0,),
            entities=entities,
            relations=(),
            has_labels=False,
            metadata={"language": str(record.get("lang") or "")},
        )

    def parse_all(self, records) -> list[Document]:
        """Convert an iterable of raw records.

        Args:
            records: Iterable of raw MultiNERD objects.

        Returns:
            A list of parsed documents in input order.
        """
        return [self.parse(record, index) for index, record in enumerate(records)]

    def _parse_entities(
        self, words: list[str], tags: list[int]
    ) -> tuple[Entity, ...]:
        """Reconstruct entity spans from BIO tag indices.

        Each span becomes its own single-mention entity. MultiNERD annotates no
        coreference, so grouping repeated surface forms here would invent links
        the corpus never asserted.

        Args:
            words: Tokenised sentence.
            tags: Tag index per token.

        Returns:
            One entity per annotated span, in reading order.
        """
        entities: list[Entity] = []
        start: int | None = None
        active = ""

        for position in range(len(words)):
            tag = (
                MULTINERD_TAGS[tags[position]]
                if position < len(tags) and 0 <= tags[position] < len(MULTINERD_TAGS)
                else "O"
            )
            prefix, _, raw_type = tag.partition("-")

            if prefix == "B" or (prefix == "I" and raw_type != active):
                if start is not None:
                    entities.append(self._build(words, start, position, active))
                start = position
                active = raw_type
            elif prefix == "O" and start is not None:
                entities.append(self._build(words, start, position, active))
                start = None
                active = ""

        if start is not None:
            entities.append(self._build(words, start, len(words), active))
        return tuple(entities)

    def _build(
        self, words: list[str], start: int, end: int, raw_type: str
    ) -> Entity:
        """Assemble an entity from a word range.

        Args:
            words: Tokenised sentence.
            start: Inclusive word index.
            end: Exclusive word index.
            raw_type: MultiNERD type name.

        Returns:
            The entity, holding a single mention.
        """
        return Entity(
            entity_type=canonical_entity_type(raw_type, MULTINERD_TYPE_MAP),
            mentions=(
                Mention(
                    text=" ".join(words[start:end]),
                    start=start,
                    end=end,
                    sentence_id=0,
                ),
            ),
        )
