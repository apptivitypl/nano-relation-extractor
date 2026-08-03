"""Structures describing what the model extracted from a text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictedMention:
    """One surface occurrence found by the NER head.

    Attributes:
        text: Surface form as it appears in the input.
        start: Inclusive word index in the tokenised input.
        end: Exclusive word index in the tokenised input.
    """

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class PredictedEntity:
    """A cluster of mentions treated as one entity.

    Attributes:
        index: Position of this entity in the extraction result.
        name: Representative surface form, taken from the first mention.
        entity_type: Predicted coarse type such as ``PER`` or ``ORG``.
        mentions: Every occurrence assigned to this cluster.
    """

    index: int
    name: str
    entity_type: str
    mentions: tuple[PredictedMention, ...]

    @property
    def mention_count(self) -> int:
        """Number of surface occurrences in the cluster."""
        return len(self.mentions)


@dataclass(frozen=True)
class PredictedRelation:
    """A relation the model asserts between two entities.

    Attributes:
        head: Index of the subject entity.
        tail: Index of the object entity.
        relation: Relation identifier such as ``P17``.
        label: Human readable relation name.
        confidence: Model confidence, above ``0.5`` for predicted relations.
    """

    head: int
    tail: int
    relation: str
    label: str
    confidence: float


@dataclass(frozen=True)
class ExtractionResult:
    """Everything the model extracted from one input text.

    Attributes:
        words: Tokenised input, truncated to the model's window.
        entities: Predicted entity clusters.
        relations: Predicted relations, ordered by descending confidence.
        truncated_words: Number of input words dropped by truncation.
    """

    words: tuple[str, ...]
    entities: tuple[PredictedEntity, ...]
    relations: tuple[PredictedRelation, ...]
    truncated_words: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the result."""
        return {
            "entities": [
                {
                    "index": entity.index,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "mentions": [
                        {
                            "text": mention.text,
                            "start": mention.start,
                            "end": mention.end,
                        }
                        for mention in entity.mentions
                    ],
                }
                for entity in self.entities
            ],
            "relations": [
                {
                    "head": relation.head,
                    "head_name": self.entities[relation.head].name,
                    "tail": relation.tail,
                    "tail_name": self.entities[relation.tail].name,
                    "relation": relation.relation,
                    "label": relation.label,
                    "confidence": relation.confidence,
                }
                for relation in self.relations
            ],
            "truncated_words": self.truncated_words,
        }

    def render(self) -> str:
        """Return a human readable summary of the extraction.

        Returns:
            A plain text report of entities and relations.
        """
        lines: list[str] = []
        if self.truncated_words:
            lines.append(
                f"Note: {self.truncated_words} words were truncated to fit the "
                "model window."
            )
            lines.append("")

        if not self.entities:
            lines.append("No entities found.")
            return "\n".join(lines)

        lines.append(f"Entities ({len(self.entities)})")
        for entity in self.entities:
            occurrences = (
                f"  x{entity.mention_count}" if entity.mention_count > 1 else ""
            )
            lines.append(
                f"  [{entity.index}] {entity.entity_type:<5} {entity.name}{occurrences}"
            )

        lines.append("")
        if not self.relations:
            lines.append("Relations (0)")
            lines.append("  none predicted")
            return "\n".join(lines)

        lines.append(f"Relations ({len(self.relations)})")
        for relation in self.relations:
            head = self.entities[relation.head].name
            tail = self.entities[relation.tail].name
            lines.append(
                f"  {head} --[{relation.label}]--> {tail}"
                f"   ({relation.confidence:.2f})"
            )
        return "\n".join(lines)
