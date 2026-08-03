"""Raw corpus retrieval from the Hugging Face Hub.

``thunlp/docred`` is distributed as a loading script plus gzipped JSON archives
and was never converted to Parquet. Script execution was removed in
``datasets`` 3.0, so :func:`datasets.load_dataset` cannot read it on any current
release. This module downloads the archives directly and rebuilds
:class:`datasets.Dataset` objects, which keeps the familiar API while avoiding
``trust_remote_code`` entirely.

The corpus is public, so no credential is needed. The optional ``token``
argument exists for private mirrors; when it is ``None``, ``huggingface_hub``
resolves a token itself from ``HF_TOKEN`` or the CLI login cache if either is
present, and otherwise downloads anonymously.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from datasets import Dataset
from huggingface_hub import hf_hub_download

RELATION_INFO_FILENAME = "data/rel_info.json.gz"

SPLIT_FILENAMES: dict[str, str] = {
    "train_annotated": "data/train_annotated.json.gz",
    "train_distant": "data/train_distant.json.gz",
    "dev": "data/dev.json.gz",
    "test": "data/test.json.gz",
}


@runtime_checkable
class DocumentSource(Protocol):
    """Provides raw corpus records and the relation vocabulary."""

    def load_split(self, split: str, limit: int | None = None) -> Dataset:
        """Return the raw records of a split.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Optional cap on the number of returned records.

        Returns:
            A :class:`datasets.Dataset` of raw records.
        """
        ...

    def load_relation_info(self) -> dict[str, str]:
        """Return the mapping of relation identifier to human readable name."""
        ...


class DocREDHubSource:
    """Downloads DocRED archives from the Hub and decodes them in memory.

    Args:
        repo_id: Dataset repository identifier.
        token: Optional Hub token for private mirrors. Leave unset for the
            public corpus.
        cache_dir: Optional download cache directory.
    """

    def __init__(
        self,
        repo_id: str = "thunlp/docred",
        token: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._repo_id = repo_id
        self._token = token
        self._cache_dir = str(cache_dir) if cache_dir is not None else None

    @property
    def repo_id(self) -> str:
        """Dataset repository this source reads from."""
        return self._repo_id

    def available_splits(self) -> tuple[str, ...]:
        """Return every split name this source can serve."""
        return tuple(SPLIT_FILENAMES)

    def load_split(self, split: str, limit: int | None = None) -> Dataset:
        """Download and decode a split into a :class:`datasets.Dataset`.

        Args:
            split: One of ``train_annotated``, ``train_distant``, ``dev``, ``test``.
            limit: Optional cap on the number of returned records.

        Returns:
            A dataset whose records mirror the original DocRED JSON objects.

        Raises:
            KeyError: If the split name is unknown.
        """
        if split not in SPLIT_FILENAMES:
            raise KeyError(
                f"Unknown split {split!r}. Available splits: "
                f"{', '.join(SPLIT_FILENAMES)}."
            )
        records = self._read_gzipped_json(SPLIT_FILENAMES[split])
        if limit is not None:
            records = records[:limit]
        return Dataset.from_list(records)

    def load_relation_info(self) -> dict[str, str]:
        """Download and decode the relation description mapping.

        Returns:
            Mapping of relation identifier to human readable name.
        """
        return dict(self._read_gzipped_json(RELATION_INFO_FILENAME))

    def _read_gzipped_json(self, filename: str):
        """Fetch a gzipped JSON file from the Hub and decode it.

        Args:
            filename: Path of the file inside the dataset repository.

        Returns:
            The decoded JSON payload.
        """
        local_path = hf_hub_download(
            repo_id=self._repo_id,
            filename=filename,
            repo_type="dataset",
            token=self._token,
            cache_dir=self._cache_dir,
        )
        with gzip.open(local_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
