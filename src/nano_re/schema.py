"""Label vocabularies shared by every pipeline stage.

The schema is the single source of truth for both task label spaces. Training,
evaluation, ONNX export and the generated model card all read the same object,
so head widths and decoded label names can never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

OUTSIDE_TAG = "O"
NA_RELATION = "NA"

DOCRED_ENTITY_TYPES: tuple[str, ...] = ("PER", "ORG", "LOC", "TIME", "NUM", "MISC")
"""Entity types annotated in DocRED, fixed by the dataset specification."""


@dataclass(frozen=True)
class LabelSchema:
    """Bidirectional label vocabularies for the NER and relation tasks.

    The relation vocabulary always reserves index ``0`` for :data:`NA_RELATION`.
    That slot doubles as the adaptive threshold class used by the relation loss,
    so its position is part of the contract rather than an implementation detail.

    Attributes:
        entity_types: Ordered entity type names used to build BIO tags.
        relation_ids: Ordered Wikidata property identifiers, excluding ``NA``.
        relation_names: Human readable description for each relation identifier.
    """

    entity_types: tuple[str, ...]
    relation_ids: tuple[str, ...]
    relation_names: dict[str, str]

    @classmethod
    def from_relation_info(
        cls,
        relation_info: dict[str, str],
        entity_types: tuple[str, ...] = DOCRED_ENTITY_TYPES,
    ) -> "LabelSchema":
        """Build a schema from the dataset's relation description mapping.

        Args:
            relation_info: Mapping of relation identifier to human readable name.
            entity_types: Ordered entity types used for the BIO vocabulary.

        Returns:
            A :class:`LabelSchema` with deterministically ordered vocabularies.
        """
        relation_ids = tuple(sorted(relation_info))
        return cls(
            entity_types=tuple(entity_types),
            relation_ids=relation_ids,
            relation_names=dict(relation_info),
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
            relation_id: Wikidata property identifier such as ``P17``.

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
        )
