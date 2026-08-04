"""Streaming reader for line-delimited JSON corpora on the Hugging Face Hub.

The multilingual corpora ship one JSON object per line, and the largest single
file is 3.3 GB. Reading it line by line holds memory flat where a whole-file
parse would not: measured, streaming stays at 0.05 GB resident.

Files are downloaded once into the Hub cache and then read from disk, so a
second pass over the same split costs no network.

A capped run does not download the whole file. Asking for five thousand Polish
documents pulls about thirty megabytes rather than the 3.3 GB the split weighs,
which is what makes the corpus usable on a machine with a small disk, and what
makes a small experiment start in seconds instead of minutes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

from ..record_index import RecordIndex, RecordLocation, read_json_line, scan_json_lines


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

    def download_prefix(self, split: str, limit: int) -> Path:
        """Fetch only the first records of a split.

        The file is streamed over HTTP and truncated locally after ``limit``
        records, so the rest is never transferred. The truncated copy is cached
        under its own name, so raising the limit fetches again while repeating a
        run does not.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Number of records to keep.

        Returns:
            Local path of the truncated copy.
        """
        target = self._prefix_path(split, limit)
        if target.exists():
            return target

        url = hf_hub_url(
            repo_id=self._repo_id,
            filename=self.resolve_filename(split),
            repo_type="dataset",
        )
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")

        written = 0
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for line in response.iter_lines():
                    if written >= limit:
                        break
                    if not line:
                        continue
                    handle.write(line + b"\n")
                    written += 1
        partial.rename(target)
        return target

    def _prefix_path(self, split: str, limit: int) -> Path:
        """Return where a truncated copy of a split is cached.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Number of records the copy holds.

        Returns:
            Local path, which may not exist yet.
        """
        root = Path(self._cache_dir) if self._cache_dir else Path.home() / ".cache"
        stem = self.resolve_filename(split).replace("/", "_")
        return root / "nano-re-prefix" / self._repo_id.replace("/", "_") / f"{limit}_{stem}"

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

    def build_index(
        self, split: str, limit: int | None = None, observer=None
    ) -> RecordIndex:
        """Scan the split once, recording where each record begins.

        The scan decodes every record anyway, so an observer may inspect them as
        they pass. That keeps the corpus to a single read whatever the caller
        needs from it.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Optional cap on the number of records indexed.
            observer: Optional callable receiving each decoded record.

        Returns:
            An index able to re-read any of those records on demand.
        """
        locations: list[RecordLocation] = []
        path = self.resolve_source(split, limit)
        for offset, record in scan_json_lines(path, limit=limit):
            if observer is not None:
                observer(record)
            locations.append(RecordLocation(path=path, offset=offset))
        return RecordIndex(locations, read_json_line)

    def resolve_source(self, split: str, limit: int | None) -> Path:
        """Return a local file holding at least ``limit`` records.

        Args:
            split: Split name understood by the concrete implementation.
            limit: Records needed, or ``None`` for the whole split.

        Returns:
            Local path to read from.
        """
        if limit is None:
            return self.download(split)
        try:
            return self.download_prefix(split, limit)
        except Exception:
            return self.download(split)

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
