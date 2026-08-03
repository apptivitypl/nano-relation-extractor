"""Composition of several corpora into one training stream.

The relation corpus and the entity corpus differ by an order of magnitude in
size and supervise different heads. Concatenating them would let the larger one
dominate whole epochs; interleaving them by weight keeps both heads receiving
signal throughout training.

Records are materialised as parsed documents rather than as tensors. A parsed
document is far smaller than its encoded form, which is what lets a corpus of a
hundred thousand records participate without the encoded footprint that made
eager encoding impossible.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterator

from torch.utils.data import Dataset

from .document import Document
from .encoder import DocumentEncoder, EncodedDocument


@dataclass(frozen=True)
class CorpusSpec:
    """One corpus contributing to a training stream.

    Attributes:
        name: Short identifier used in reports.
        source: Reader streaming raw records.
        parser: Converts a raw record into a document.
        split: Split name passed to the reader.
        weight: Relative sampling weight against the other corpora.
        limit: Optional cap on records taken from this corpus.
    """

    name: str
    source: object
    parser: object
    split: str
    weight: float = 1.0
    limit: int | None = None


@dataclass(frozen=True)
class CorpusStatistics:
    """Counters describing one corpus's contribution.

    Attributes:
        name: Corpus identifier.
        documents: Documents taken from the corpus.
        entities: Annotated entities across those documents.
        relations: Annotated relation triples, zero for entity-only corpora.
        supervises_relations: Whether the corpus trains the relation head.
        languages: Languages observed in the sampled documents.
    """

    name: str
    documents: int
    entities: int
    relations: int
    supervises_relations: bool
    languages: tuple[str, ...]

    def describe(self) -> str:
        """Return a one-line human readable summary."""
        role = "NER+RE" if self.supervises_relations else "NER only"
        languages = ",".join(self.languages) if self.languages else "?"
        return (
            f"  {self.name:<10} {self.documents:>7} docs  "
            f"{self.entities:>8} entities  {self.relations:>7} relations  "
            f"[{role}]  {languages}"
        )


class MultiCorpusDataset(Dataset):
    """Interleaves parsed documents from several corpora and encodes on demand.

    The encoder arrives as a factory rather than as an instance because the
    label schema it needs is derived from the very documents this dataset
    parses. Resolving it on first access breaks that circular dependency without
    forcing the caller to sequence two passes by hand.

    Args:
        specs: Corpora to draw from.
        encoder_factory: Produces the encoder once parsing has completed.
        sample_negatives: Whether to subsample negative pairs.
        cache: Whether to retain encoded documents between epochs.
    """

    def __init__(
        self,
        specs: list[CorpusSpec],
        encoder_factory: Callable[[], DocumentEncoder],
        sample_negatives: bool,
        cache: bool = False,
    ) -> None:
        self._encoder_factory = encoder_factory
        self._encoder: DocumentEncoder | None = None
        self._sample_negatives = sample_negatives
        self._cache: dict[int, EncodedDocument | None] | None = {} if cache else None
        self._documents: list[Document] = []
        self._statistics: list[CorpusStatistics] = []
        self._load(specs)

    @property
    def encoder(self) -> DocumentEncoder:
        """Encoder for this stream, resolved after parsing completed."""
        if self._encoder is None:
            self._encoder = self._encoder_factory()
        return self._encoder

    @property
    def statistics(self) -> list[CorpusStatistics]:
        """Per-corpus counters gathered while loading."""
        return list(self._statistics)

    @property
    def documents(self) -> list[Document]:
        """Parsed documents in interleaved order."""
        return self._documents

    def __len__(self) -> int:
        """Number of documents in the stream."""
        return len(self._documents)

    def __getitem__(self, index: int) -> EncodedDocument | None:
        """Encode one document.

        Args:
            index: Position in the interleaved stream.

        Returns:
            The encoded document, or ``None`` when it yields no usable tensors.
        """
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        encoded = self.encoder.encode(
            self._documents[index], sample_negatives=self._sample_negatives
        )
        if self._cache is not None:
            self._cache[index] = encoded
        return encoded

    def _load(self, specs: list[CorpusSpec]) -> None:
        """Parse every weighted corpus and interleave the results.

        A corpus given a weight of zero is skipped before it is read, not after.
        That matters beyond saving a download: parsing populates the relation
        inventory that sizes the model's head, so a corpus excluded for licensing
        reasons must not leave its predicates behind in the label schema.

        Args:
            specs: Corpora to draw from.
        """
        per_corpus: list[list[Document]] = []
        active = [spec for spec in specs if spec.weight > 0]
        for spec in active:
            documents = list(self._parse(spec))
            per_corpus.append(documents)
            languages = sorted(
                {
                    document.metadata.get("language", "")
                    for document in documents
                    if document.metadata.get("language")
                }
            )
            self._statistics.append(
                CorpusStatistics(
                    name=spec.name,
                    documents=len(documents),
                    entities=sum(document.num_entities for document in documents),
                    relations=sum(len(document.relations) for document in documents),
                    supervises_relations=bool(
                        getattr(spec.source, "provides_relations", True)
                    ),
                    languages=tuple(languages),
                )
            )
        self._documents = self._interleave(per_corpus, [s.weight for s in active])

    def _parse(self, spec: CorpusSpec) -> Iterator[Document]:
        """Stream and parse one corpus.

        Args:
            spec: Corpus to read.

        Yields:
            Parsed documents.
        """
        for index, record in enumerate(
            spec.source.iter_records(spec.split, limit=spec.limit)
        ):
            yield spec.parser.parse(record, index)

    def _interleave(
        self, per_corpus: list[list[Document]], weights: list[float]
    ) -> list[Document]:
        """Merge corpora into one stream honouring the requested weights.

        Args:
            per_corpus: Documents grouped by corpus.
            weights: Relative sampling weight per corpus.

        A corpus given a non-positive weight contributes nothing and is dropped
        up front, rather than sitting in the rotation forever without ever being
        due.

        Merging is by virtual time: a corpus with weight ``w`` places its ``k``
        th document at ``k / w``, and documents are emitted in that order. Every
        step consumes exactly one document, so the merge always terminates and
        the realised ratio matches the requested weights regardless of how the
        corpus sizes differ.

        Args:
            per_corpus: Documents grouped by corpus.
            weights: Relative sampling weight per corpus.

        Returns:
            The interleaved document list.
        """
        active = [
            (documents, weight)
            for documents, weight in zip(per_corpus, weights)
            if weight > 0 and documents
        ]
        if not active:
            return []

        total = sum(weight for _, weight in active)
        queue = [
            (1.0 / (weight / total), index, 0) for index, (_, weight) in enumerate(active)
        ]
        heapq.heapify(queue)
        merged: list[Document] = []

        while queue:
            _, index, position = heapq.heappop(queue)
            documents, weight = active[index]
            merged.append(documents[position])
            if position + 1 < len(documents):
                share = weight / total
                heapq.heappush(queue, ((position + 2) / share, index, position + 1))
        return merged
