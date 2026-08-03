"""Reader for the Re-DocRED document-level relation corpus.

Re-DocRED is the revised release of DocRED, and the revision matters: the
original is missing a large share of its true positives, so a model trained
against it is punished for correct predictions. Measured on the development
split, Re-DocRED carries 34.6 gold triples per document against DocRED's 12.3.

It is English only, and it is here for exactly that reason: it is the strongest
human-annotated document-level relation supervision available, and it is MIT
licensed, so it constrains the release less than anything else in the stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .jsonl import JsonlHubSource

SPLIT_FILES: dict[str, str] = {
    "train": "train_revised.json",
    "dev": "dev_revised.json",
    "test": "test_revised.json",
}


class ReDocredSource(JsonlHubSource):
    """Streams Re-DocRED, which is a JSON array rather than JSON Lines.

    Args:
        languages: Requested languages, used only to report what is uncovered.
        cache_dir: Optional download cache directory.
        token: Optional Hub token.
    """

    def __init__(
        self,
        languages: tuple[str, ...] = ("en",),
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(
            repo_id="tonytan48/Re-DocRED",
            name="redocred",
            provides_relations=True,
            cache_dir=cache_dir,
            token=token,
        )
        self._requested = languages

    @property
    def languages(self) -> tuple[str, ...]:
        """Languages this reader provides."""
        return ("en",)

    @property
    def uncovered_languages(self) -> tuple[str, ...]:
        """Requested languages Re-DocRED does not provide."""
        return tuple(language for language in self._requested if language != "en")

    def resolve_filename(self, split: str) -> str:
        """Return the repository path of a split's file.

        Args:
            split: Split name, optionally carrying a language suffix.

        Returns:
            Path of the file inside the dataset repository.
        """
        name = split.partition(":")[0]
        return SPLIT_FILES.get(name, SPLIT_FILES["train"])

    def iter_records(self, split: str, limit: int | None = None) -> Iterator[dict]:
        """Yield decoded records from the split's JSON array.

        The largest file is a few tens of megabytes, so it is read whole rather
        than streamed; the streaming machinery exists for the multi-gigabyte
        corpora, not for this one.

        Args:
            split: Split name.
            limit: Optional cap on the number of records yielded.

        Yields:
            Raw DocRED-shaped records.
        """
        path = self.download(split)
        records = json.loads(path.read_text(encoding="utf-8"))
        for index, record in enumerate(records):
            if limit is not None and index >= limit:
                return
            record.setdefault("lan", "en")
            yield record
