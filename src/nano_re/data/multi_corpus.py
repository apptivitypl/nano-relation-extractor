"""Composition of several corpora into one training stream.

The relation corpus and the entity corpus differ by an order of magnitude in
size and supervise different heads. Concatenating them would let the larger one
dominate whole epochs; interleaving them by weight keeps both heads receiving
signal throughout training.

Nothing is materialised. A corpus that can locate its records by byte offset is
read one record at a time, at the moment the trainer asks for it: the eight
language training split holds 5.6 million documents, and keeping their parsed
form in memory would need about 57 GB. Indexing them costs tens of megabytes
instead.

Building that index reads every file once, and the same pass feeds the relation
inventory and the corpus counters, so the corpus is read once rather than once
per purpose. A corpus too small or too awkward to index, such as one shipped as
a single JSON array, falls back to being held in memory.
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
        self._specs: list[CorpusSpec] = []
        self._stores: list[object] = []
        self._order: list[tuple[int, int]] = []
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

    def document_at(self, index: int) -> Document | None:
        """Parse the document at a position in the stream.

        Args:
            index: Position in the interleaved stream.

        Returns:
            The parsed document, or ``None`` when its record cannot be read.
        """
        corpus, local = self._order[index]
        spec = self._specs[corpus]
        store = self._stores[corpus]
        if isinstance(store, list):
            return store[local]
        record = store.read(local)
        if record is None:
            return None
        return spec.parser.parse(record, local)

    def documents(self) -> Iterator[Document]:
        """Yield every document in stream order, one at a time.

        Materialising the stream would defeat the purpose of indexing it, so
        callers that need every document receive an iterator rather than a list.

        Yields:
            Parsed documents.
        """
        for index in range(len(self)):
            document = self.document_at(index)
            if document is not None:
                yield document

    def __len__(self) -> int:
        """Number of documents in the stream."""
        return len(self._order)

    def __getitem__(self, index: int) -> EncodedDocument | None:
        """Encode one document.

        Args:
            index: Position in the interleaved stream.

        Returns:
            The encoded document, or ``None`` when it yields no usable tensors.
        """
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        document = self.document_at(index)
        encoded = (
            None
            if document is None
            else self.encoder.encode(
                document, sample_negatives=self._sample_negatives
            )
        )
        if self._cache is not None:
            self._cache[index] = encoded
        return encoded

    def _index_corpus(self, spec: CorpusSpec) -> tuple[object, CorpusStatistics]:
        """Index one corpus and collect its counters in the same pass.

        Args:
            spec: Corpus to read.

        Returns:
            Either a record index or a list of parsed documents, together with
            the statistics observed while reading.
        """
        seen_languages: set[str] = set()
        counters = {"documents": 0, "entities": 0, "relations": 0}

        def observe(record: dict) -> None:
            document = spec.parser.parse(record, counters["documents"])
            counters["documents"] += 1
            counters["entities"] += document.num_entities
            counters["relations"] += len(document.relations)
            language = document.metadata.get("language", "")
            if language:
                seen_languages.add(language)

        builder = getattr(spec.source, "build_index", None)
        if builder is not None and callable(builder):
            store: object = builder(spec.split, limit=spec.limit, observer=observe)
        else:
            documents = [
                spec.parser.parse(record, index)
                for index, record in enumerate(
                    spec.source.iter_records(spec.split, limit=spec.limit)
                )
            ]
            counters["documents"] = len(documents)
            counters["entities"] = sum(item.num_entities for item in documents)
            counters["relations"] = sum(len(item.relations) for item in documents)
            seen_languages.update(
                item.metadata.get("language", "")
                for item in documents
                if item.metadata.get("language")
            )
            store = documents

        indexed = len(store) if not isinstance(store, list) else len(store)
        return store, CorpusStatistics(
            name=spec.name,
            documents=indexed,
            entities=counters["entities"],
            relations=counters["relations"],
            supervises_relations=bool(
                getattr(spec.source, "provides_relations", True)
            ),
            languages=tuple(sorted(seen_languages)),
        )

    def _load(self, specs: list[CorpusSpec]) -> None:
        """Parse every weighted corpus and interleave the results.

        A corpus given a weight of zero is skipped before it is read, not after.
        That matters beyond saving a download: parsing populates the relation
        inventory that sizes the model's head, so a corpus excluded for licensing
        reasons must not leave its predicates behind in the label schema.

        Args:
            specs: Corpora to draw from.
        """
        active = [spec for spec in specs if spec.weight > 0]
        self._specs = active
        self._stores = []
        per_corpus: list[list[tuple[int, int]]] = []

        for position, spec in enumerate(active):
            store, statistics = self._index_corpus(spec)
            self._stores.append(store)
            self._statistics.append(statistics)
            per_corpus.append(
                [(position, local) for local in range(statistics.documents)]
            )

        self._order = self._interleave(per_corpus, [s.weight for s in active])

    def _interleave(self, per_corpus: list[list], weights: list[float]) -> list:
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
            (items, weight)
            for items, weight in zip(per_corpus, weights)
            if weight > 0 and items
        ]
        if not active:
            return []

        total = sum(weight for _, weight in active)
        queue = [
            (1.0 / (weight / total), index, 0) for index, (_, weight) in enumerate(active)
        ]
        heapq.heapify(queue)
        merged: list = []

        while queue:
            _, index, position = heapq.heappop(queue)
            items, weight = active[index]
            merged.append(items[position])
            if position + 1 < len(items):
                share = weight / total
                heapq.heappush(queue, ((position + 2) / share, index, position + 1))
        return merged
