"""Dataset agnostic document representation.

These structures decouple every downstream stage from any corpus's layout.
Supporting another corpus means writing a new parser that emits :class:`Document`
objects; the encoder, model and metrics stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mention:
    """A single surface occurrence of an entity inside a document.

    Attributes:
        text: Surface form as annotated in the corpus.
        start: Inclusive word index within the flattened document.
        end: Exclusive word index within the flattened document.
        sentence_id: Index of the sentence containing the mention.
    """

    text: str
    start: int
    end: int
    sentence_id: int


@dataclass(frozen=True)
class Entity:
    """A coreference cluster of mentions referring to the same real world item.

    Attributes:
        entity_type: Coarse type such as ``PER`` or ``ORG``.
        mentions: Every surface occurrence belonging to this cluster.
    """

    entity_type: str
    mentions: tuple[Mention, ...]

    @property
    def name(self) -> str:
        """Surface form of the first mention, used for readable predictions."""
        return self.mentions[0].text if self.mentions else ""


@dataclass(frozen=True)
class RelationTriple:
    """A directed relation between two entity clusters.

    Attributes:
        head: Index of the subject entity within :attr:`Document.entities`.
        tail: Index of the object entity within :attr:`Document.entities`.
        relation: Relation identifier such as ``P17``.
    """

    head: int
    tail: int
    relation: str


@dataclass(frozen=True)
class Document:
    """A parsed document with entity and relation annotations.

    Attributes:
        doc_id: Stable identifier, used to key evaluation triples.
        words: Flattened whitespace tokens of the whole document.
        sentence_offsets: Word index at which each sentence starts.
        entities: Annotated entity clusters.
        relations: Gold relation triples. Empty for unlabelled splits.
        has_labels: Whether relation annotations were present in the source.
    """

    doc_id: str
    words: tuple[str, ...]
    sentence_offsets: tuple[int, ...]
    entities: tuple[Entity, ...] = ()
    relations: tuple[RelationTriple, ...] = ()
    has_labels: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def num_words(self) -> int:
        """Number of whitespace tokens in the document."""
        return len(self.words)

    @property
    def num_entities(self) -> int:
        """Number of annotated entity clusters."""
        return len(self.entities)
