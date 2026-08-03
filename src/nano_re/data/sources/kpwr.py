"""Reader for the KPWr Polish named entity corpus.

KPWr replaces MultiNERD as the source of extra entity supervision. MultiNERD is
larger and multilingual, but it is licensed CC BY-NC-SA: training on it would
produce a model nobody could use commercially, which defeats the purpose of an
open release. KPWr is CC BY-3.0 and imposes no such restriction.

The corpus ships as IOB text rather than JSON, one token per line with the tag
in the final column, and blank lines between sentences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..record_index import RecordIndex, RecordLocation
from .jsonl import JsonlHubSource

DOCUMENT_MARKER = "-DOCSTART"

SPLIT_FILES: dict[str, str] = {
    "train": "data/kpwr-ner-n82-train-tune.iob",
    "dev": "data/kpwr-ner-n82-test.iob",
    "test": "data/kpwr-ner-n82-test.iob",
}


class KpwrSource(JsonlHubSource):
    """Streams the KPWr corpus as one record per sentence.

    Args:
        languages: Requested languages. KPWr is Polish only; other languages are
            reported through :attr:`uncovered_languages` rather than silently
            ignored.
        cache_dir: Optional download cache directory.
        token: Optional Hub token.
    """

    def __init__(
        self,
        languages: tuple[str, ...] = ("pl",),
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(
            repo_id="clarin-pl/kpwr-ner",
            name="kpwr",
            provides_relations=False,
            cache_dir=cache_dir,
            token=token,
        )
        self._requested = languages

    @property
    def languages(self) -> tuple[str, ...]:
        """Languages this reader provides."""
        return ("pl",)

    @property
    def uncovered_languages(self) -> tuple[str, ...]:
        """Requested languages KPWr does not provide."""
        return tuple(language for language in self._requested if language != "pl")

    def resolve_filename(self, split: str) -> str:
        """Return the repository path of a split's file.

        Args:
            split: Split name, optionally carrying a language suffix.

        Returns:
            Path of the file inside the dataset repository.
        """
        name = split.partition(":")[0]
        return SPLIT_FILES.get(name, SPLIT_FILES["train"])

    def build_index(
        self, split: str, limit: int | None = None, observer=None
    ) -> RecordIndex:
        """Scan the file once, recording where each sentence begins.

        Args:
            split: Split name.
            limit: Optional cap on the number of sentences indexed.
            observer: Optional callable receiving each decoded sentence.

        Returns:
            An index able to re-read any sentence on demand.
        """
        path = self.download(split)
        locations: list[RecordLocation] = []
        pending: int | None = None

        with path.open("rb") as handle:
            while True:
                if limit is not None and len(locations) >= limit:
                    break
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if line.startswith(DOCUMENT_MARKER):
                    continue
                if not line.strip():
                    pending = None
                    continue
                if pending is None:
                    pending = offset
                    locations.append(RecordLocation(path=path, offset=offset))

        if observer is not None:
            index = RecordIndex(locations, _read_sentence)
            for position in range(len(index)):
                record = index.read(position)
                if record is not None:
                    observer(record)
            index.close()
        return RecordIndex(locations, _read_sentence)

    def iter_records(self, split: str, limit: int | None = None) -> Iterator[dict]:
        """Yield one record per sentence.

        Args:
            split: Split name.
            limit: Optional cap on the number of records yielded.

        Yields:
            Records with ``tokens``, ``tags`` and ``lang``.
        """
        path = self.download(split)
        tokens: list[str] = []
        tags: list[str] = []
        emitted = 0

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if stripped.startswith(DOCUMENT_MARKER):
                    continue
                if not stripped.strip():
                    if tokens:
                        if limit is not None and emitted >= limit:
                            return
                        emitted += 1
                        yield {"tokens": tokens, "tags": tags, "lang": "pl"}
                        tokens, tags = [], []
                    continue
                columns = stripped.split("\t")
                if len(columns) < 2:
                    continue
                tokens.append(columns[0])
                tags.append(columns[-1])

        if tokens and (limit is None or emitted < limit):
            yield {"tokens": tokens, "tags": tags, "lang": "pl"}


def _read_sentence(handle) -> dict | None:
    """Read one sentence from a handle positioned at its first token.

    Args:
        handle: Binary file handle positioned at the sentence start.

    Returns:
        A record with ``tokens``, ``tags`` and ``lang``, or ``None`` at the end
        of the file.
    """
    tokens: list[str] = []
    tags: list[str] = []
    while True:
        raw = handle.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if line.startswith(DOCUMENT_MARKER):
            continue
        if not line.strip():
            break
        columns = line.split("\t")
        if len(columns) < 2:
            continue
        tokens.append(columns[0])
        tags.append(columns[-1])
    if not tokens:
        return None
    return {"tokens": tokens, "tags": tags, "lang": "pl"}
