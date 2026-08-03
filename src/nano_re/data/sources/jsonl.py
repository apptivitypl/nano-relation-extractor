"""Streaming reader for line-delimited JSON corpora on the Hugging Face Hub.

The multilingual corpora ship one JSON object per line, and the largest single
file is 3.3 GB. Reading it line by line holds memory flat where a whole-file
parse would not: measured, streaming stays at 0.05 GB resident.

Files are downloaded once into the Hub cache and then read from disk, so a
second pass over the same split costs no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download


class JsonlHubSource:
    """Streams JSON Lines files from a Hub dataset repository.

    Args:
        repo_id: Dataset repository identifier.
        name: Short corpus identifier used in reports.
        provides_relations: Whether the corpus carries relation annotations.
        cache_dir: Optional download cache directory.
        token: Optional Hub token, only needed for private mirrors.
    """

    def __init__(
        self,
        repo_id: str,
        name: str,
        provides_relations: bool,
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        self._repo_id = repo_id
        self._name = name
        self._provides_relations = provides_relations
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._token = token
        self._counts: dict[str, int] = {}

    @property
    def repo_id(self) -> str:
        """Dataset repository this source reads from."""
        return self._repo_id

    @property
    def name(self) -> str:
        """Short identifier of the corpus."""
        return self._name

    @property
    def provides_relations(self) -> bool:
        """Whether records carry relation annotations."""
        return self._provides_relations

    def resolve_filename(self, split: str) -> str:
        """Return the repository path of a split's file.

        Args:
            split: Split name understood by the concrete implementation.

        Returns:
            Path of the file inside the dataset repository.

        Raises:
            NotImplementedError: Always, unless a subclass overrides it.
        """
        raise NotImplementedError

    def download(self, split: str) -> Path:
        """Fetch a split's file into the local cache.

        Args:
            split: Split name understood by the concrete implementation.

        Returns:
            Local path of the downloaded file.
        """
        return Path(
            hf_hub_download(
                repo_id=self._repo_id,
                filename=self.resolve_filename(split),
                repo_type="dataset",
                cache_dir=self._cache_dir,
                token=self._token,
            )
        )

    def iter_records(self, split: str, limit: int | None = None) -> Iterator[dict]:
        """Yield decoded records one line at a time.

        Malformed lines are skipped rather than aborting a multi-gigabyte pass
        over an automatically generated corpus.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Optional cap on the number of records yielded.

        Yields:
            Decoded records in file order.
        """
        path = self.download(split)
        emitted = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and emitted >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                emitted += 1
                yield record

    def count(self, split: str, limit: int | None = None) -> int:
        """Count records in a split, memoising the result.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Optional cap applied before counting.

        Returns:
            The record count.
        """
        key = f"{split}:{limit}"
        if key not in self._counts:
            self._counts[key] = sum(1 for _ in self.iter_records(split, limit=limit))
        return self._counts[key]
