"""The corpus reader contract.

Every stage above this one works in terms of raw records and a relation
inventory, never in terms of a particular corpus. Supporting a new corpus means
adding a reader here and a parser beside it; nothing downstream changes.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class DocumentSource(Protocol):
    """Provides raw corpus records for a split."""

    @property
    def name(self) -> str:
        """Short identifier of the corpus, used in reports."""
        ...

    @property
    def provides_relations(self) -> bool:
        """Whether records carry relation annotations.

        A corpus without relations must not train the relation head, so this
        flag travels with the data all the way into the loss.
        """
        ...

    def iter_records(self, split: str, limit: int | None = None) -> Iterator[dict]:
        """Yield raw records one at a time.

        Streaming rather than returning a list is what keeps a multi-gigabyte
        corpus out of memory.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Optional cap on the number of records yielded.

        Yields:
            Raw records in corpus order.
        """
        ...

    def count(self, split: str, limit: int | None = None) -> int:
        """Return the number of records in a split.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Optional cap applied before counting.

        Returns:
            The record count.
        """
        ...
