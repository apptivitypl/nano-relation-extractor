"""Label vocabularies shared by every pipeline stage.

The schema is the single source of truth for both task label spaces. Training,
evaluation, ONNX export and the generated model card all read the same object,
so head widths and decoded label names can never drift apart.

Two corpora feed the model and they disagree about entity types: SREDFM uses
thirteen, MultiNERD fifteen. Rather than train on whichever happens to arrive
first, both are mapped onto one canonical inventory chosen for graph
construction: the types that become nodes worth linking. Everything else folds
into ``MISC`` instead of inflating the tag set with classes a downstream graph
would ignore.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

OUTSIDE_TAG = "O"
NA_RELATION = "NA"

CANONICAL_ENTITY_TYPES: tuple[str, ...] = (
    "PER",
    "ORG",
    "LOC",
    "DATE",
    "TIME",
    "NUMBER",
    "MEDIA",
    "EVE",
    "MISC",
)
"""Entity types the model predicts, chosen for their value as graph nodes."""

SREDFM_TYPE_MAP: dict[str, str] = {
    "PER": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "DATE": "DATE",
    "TIME": "TIME",
    "NUMBER": "NUMBER",
    "MEDIA": "MEDIA",
    "EVE": "EVE",
    "CONCEPT": "MISC",
    "MISC": "MISC",
    "CEL": "MISC",
    "DIS": "MISC",
    "UNK": "MISC",
}
"""Mapping from SREDFM entity types onto the canonical inventory."""

MULTINERD_TYPE_MAP: dict[str, str] = {
    "PER": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "TIME": "TIME",
    "MEDIA": "MEDIA",
    "EVE": "EVE",
    "ANIM": "MISC",
    "BIO": "MISC",
    "CEL": "MISC",
    "DIS": "MISC",
    "FOOD": "MISC",
    "INST": "MISC",
    "MYTH": "MISC",
    "PLANT": "MISC",
    "VEHI": "MISC",
}
"""Mapping from MultiNERD entity types onto the canonical inventory."""

MULTINERD_TAGS: tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-BIO",
    "I-BIO",
    "B-CEL",
    "I-CEL",
    "B-DIS",
    "I-DIS",
    "B-EVE",
    "I-EVE",
    "B-FOOD",
    "I-FOOD",
    "B-INST",
    "I-INST",
    "B-MEDIA",
    "I-MEDIA",
    "B-MYTH",
    "I-MYTH",
    "B-PLANT",
    "I-PLANT",
    "B-TIME",
    "I-TIME",
    "B-VEHI",
    "I-VEHI",
)
"""MultiNERD's own tag inventory, indexed exactly as the corpus encodes it."""


def canonical_entity_type(raw_type: str, mapping: dict[str, str]) -> str:
    """Fold a corpus specific entity type onto the canonical inventory.

    Args:
        raw_type: Type as the corpus spells it.
        mapping: Corpus specific mapping table.

    Returns:
        A member of :data:`CANONICAL_ENTITY_TYPES`, defaulting to ``MISC``.
    """
    return mapping.get(raw_type.strip().upper(), "MISC")


@dataclass(frozen=True)
class LabelSchema:
    """Bidirectional label vocabularies for the NER and relation tasks.

    The relation vocabulary always reserves index ``0`` for :data:`NA_RELATION`.
    That slot doubles as the adaptive threshold class used by the relation loss,
    so its position is part of the contract rather than an implementation detail.

    Attributes:
        entity_types: Ordered entity type names used to build BIO tags.
        relation_ids: Ordered relation identifiers, excluding ``NA``.
        relation_names: Human readable description for each relation identifier.
        languages: Languages the schema was built from, recorded so a bundle
            states its own scope rather than leaving callers to guess.
    """

    entity_types: tuple[str, ...]
    relation_ids: tuple[str, ...]
    relation_names: dict[str, str]
    languages: tuple[str, ...] = ()

    @classmethod
    def from_relation_info(
        cls,
        relation_info: dict[str, str],
        entity_types: tuple[str, ...] = CANONICAL_ENTITY_TYPES,
        languages: tuple[str, ...] = (),
    ) -> "LabelSchema":
        """Build a schema from a relation description mapping.

        Args:
            relation_info: Mapping of relation identifier to human readable name.
            entity_types: Ordered entity types used for the BIO vocabulary.
            languages: Languages the inventory was collected from.

        Returns:
            A :class:`LabelSchema` with deterministically ordered vocabularies.
        """
        relation_ids = tuple(sorted(relation_info))
        return cls(
            entity_types=tuple(entity_types),
            relation_ids=relation_ids,
            relation_names=dict(relation_info),
            languages=tuple(languages),
        )

    @property
    def bio_labels(self) -> tuple[str, ...]:
        """Ordered BIO tag names, starting with the outside tag."""
        tags = [OUTSIDE_TAG]
        for entity_type in self.entity_types:
            tags.append(f"B-{entity_type}")
            tags.append(f"I-{entity_type}")
        return tuple(tags)

    @property
    def relation_labels(self) -> tuple[str, ...]:
        """Ordered relation class names, with ``NA`` at index zero."""
        return (NA_RELATION,) + self.relation_ids

    @property
    def num_bio_labels(self) -> int:
        """Width of the token classification head."""
        return len(self.bio_labels)

    @property
    def num_relation_labels(self) -> int:
        """Width of the relation classification head, including ``NA``."""
        return len(self.relation_labels)

    @property
    def bio_to_id(self) -> dict[str, int]:
        """Mapping from BIO tag name to head index."""
        return {label: index for index, label in enumerate(self.bio_labels)}

    @property
    def id_to_bio(self) -> dict[int, str]:
        """Mapping from head index to BIO tag name."""
        return dict(enumerate(self.bio_labels))

    @property
    def relation_to_id(self) -> dict[str, int]:
        """Mapping from relation identifier to head index."""
        return {label: index for index, label in enumerate(self.relation_labels)}

    @property
    def id_to_relation(self) -> dict[int, str]:
        """Mapping from head index to relation identifier."""
        return dict(enumerate(self.relation_labels))

    def describe_relation(self, relation_id: str) -> str:
        """Return the human readable name of a relation identifier.

        Args:
            relation_id: Relation identifier such as ``P17``.

        Returns:
            The description if known, otherwise the identifier itself.
        """
        return self.relation_names.get(relation_id, relation_id)

    def to_dict(self) -> dict[str, object]:
        """Serialise the schema into a JSON compatible dictionary.

        Returns:
            A dictionary suitable for :func:`json.dump`.
        """
        return {
            "entity_types": list(self.entity_types),
            "relation_ids": list(self.relation_ids),
            "relation_names": self.relation_names,
            "languages": list(self.languages),
            "bio_labels": list(self.bio_labels),
            "relation_labels": list(self.relation_labels),
        }

    def save(self, path: Path) -> Path:
        """Write the schema to disk as JSON.

        Args:
            path: Destination file path.

        Returns:
            The path that was written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "LabelSchema":
        """Read a schema previously written by :meth:`save`.

        Args:
            path: Source file path.

        Returns:
            The deserialised :class:`LabelSchema`.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            entity_types=tuple(payload["entity_types"]),
            relation_ids=tuple(payload["relation_ids"]),
            relation_names=dict(payload["relation_names"]),
            languages=tuple(payload.get("languages", [])),
        )


class RelationInventory:
    """Accumulates the relation vocabulary seen across corpora and languages.

    The inventory is collected rather than hard-coded because it depends on
    which languages are in scope: a predicate frequent in one language may be
    absent from another. Counting occurrences alongside identity lets the
    training report show how much of the tail is too rare to learn, which a bare
    list would hide.
    """

    def __init__(self) -> None:
        self._names: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def add(self, relation_id: str, name: str | None = None) -> None:
        """Record one occurrence of a relation.

        Args:
            relation_id: Stable identifier, such as a Wikidata property.
            name: Optional human readable name, kept from first sighting.
        """
        self._counts[relation_id] = self._counts.get(relation_id, 0) + 1
        if name and relation_id not in self._names:
            self._names[relation_id] = name

    @property
    def counts(self) -> dict[str, int]:
        """Occurrence count per relation identifier."""
        return dict(self._counts)

    def __len__(self) -> int:
        """Number of distinct relations recorded."""
        return len(self._counts)

    def to_schema(
        self,
        languages: tuple[str, ...] = (),
        min_count: int = 1,
        max_relations: int | None = None,
    ) -> LabelSchema:
        """Build a label schema from the accumulated inventory.

        Args:
            languages: Languages the inventory was collected from.
            min_count: Smallest occurrence count kept.
            max_relations: Optional cap keeping only the most frequent
                relations. ``None`` keeps every relation meeting ``min_count``.

        Returns:
            The resulting :class:`LabelSchema`.
        """
        eligible = [
            relation_id
            for relation_id, count in self._counts.items()
            if count >= min_count
        ]
        eligible.sort(key=lambda item: (-self._counts[item], item))
        if max_relations is not None:
            eligible = eligible[:max_relations]
        return LabelSchema.from_relation_info(
            {
                relation_id: self._names.get(relation_id, relation_id)
                for relation_id in sorted(eligible)
            },
            languages=languages,
        )
