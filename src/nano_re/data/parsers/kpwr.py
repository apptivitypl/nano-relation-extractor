"""Parser for the KPWr Polish named entity corpus.

KPWr annotates 82 fine-grained categories in a ``nam_<domain>_<subtype>``
scheme. The model predicts nine, so the parser folds by domain prefix: what
matters downstream is whether a span is a person, an organisation or a place,
not which of eleven organisation subtypes it belongs to.
"""

from __future__ import annotations

from ...schema import CANONICAL_ENTITY_TYPES
from ..document import Document, Entity, Mention

PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("nam_liv", "PER"),
    ("nam_org", "ORG"),
    ("nam_loc", "LOC"),
    ("nam_fac", "LOC"),
    ("nam_eve", "EVE"),
    ("nam_pro_media", "MEDIA"),
    ("nam_pro_title", "MEDIA"),
    ("nam_num", "NUMBER"),
)
"""Ordered prefix rules folding KPWr categories onto the canonical inventory."""


def canonical_kpwr_type(raw_type: str) -> str:
    """Fold a KPWr category onto the canonical inventory.

    Args:
        raw_type: Category name such as ``nam_org_institution``.

    Returns:
        A member of :data:`CANONICAL_ENTITY_TYPES`, defaulting to ``MISC``.
    """
    lowered = raw_type.strip().lower()
    for prefix, canonical in PREFIX_MAP:
        if lowered.startswith(prefix):
            return canonical
    return "MISC"


class KpwrParser:
    """Converts KPWr sentences into the internal document representation."""

    def parse(self, record: dict, index: int) -> Document:
        """Convert a single sentence.

        Args:
            record: Raw object with ``tokens`` and ``tags``.
            index: Positional index used to build an identifier.

        Returns:
            The parsed :class:`Document`, carrying entities but no relations.
        """
        words = [str(token) for token in record.get("tokens") or []]
        tags = [str(tag) for tag in record.get("tags") or []]
        return Document(
            doc_id=f"kpwr-{index}",
            words=tuple(words),
            sentence_offsets=(0,),
            entities=self._parse_entities(words, tags),
            relations=(),
            has_labels=False,
            metadata={"language": "pl"},
        )

    def parse_all(self, records) -> list[Document]:
        """Convert an iterable of raw records.

        Args:
            records: Iterable of raw KPWr sentences.

        Returns:
            A list of parsed documents in input order.
        """
        return [self.parse(record, index) for index, record in enumerate(records)]

    def _parse_entities(
        self, words: list[str], tags: list[str]
    ) -> tuple[Entity, ...]:
        """Reconstruct entity spans from IOB tags.

        Args:
            words: Tokenised sentence.
            tags: IOB tag per token.

        Returns:
            One entity per annotated span, in reading order.
        """
        entities: list[Entity] = []
        start: int | None = None
        active = ""

        for position in range(len(words)):
            tag = tags[position] if position < len(tags) else "O"
            prefix, _, raw_type = tag.partition("-")

            if prefix == "B" or (prefix == "I" and raw_type != active):
                if start is not None:
                    entities.append(self._build(words, start, position, active))
                start = position
                active = raw_type
            elif prefix not in {"B", "I"} and start is not None:
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
            raw_type: KPWr category name.

        Returns:
            The entity, holding a single mention.
        """
        return Entity(
            entity_type=canonical_kpwr_type(raw_type),
            mentions=(
                Mention(
                    text=" ".join(words[start:end]),
                    start=start,
                    end=end,
                    sentence_id=0,
                ),
            ),
        )
