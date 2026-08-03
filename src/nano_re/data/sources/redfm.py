"""Readers for the SREDFM and REDFM relation extraction corpora.

Both share a layout: one file per language and split, each a JSON Lines stream
of documents carrying character-offset entities and subject-predicate-object
relations.

They differ in provenance, and the difference matters for honest reporting.
SREDFM is generated automatically and is large enough to train on; REDFM is the
human-filtered subset and is what evaluation numbers should come from. REDFM
covers seven languages and Polish is not among them, so a Polish deployment has
no gold evaluation available at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .jsonl import JsonlHubSource

SREDFM_LANGUAGES: tuple[str, ...] = (
    "ar", "ca", "de", "el", "en", "es", "fr", "hi", "it",
    "ja", "ko", "nl", "pl", "pt", "ru", "sv", "vi", "zh",
)
"""Languages carried by SREDFM."""

REDFM_LANGUAGES: tuple[str, ...] = ("ar", "de", "en", "es", "fr", "it", "zh")
"""Languages carried by the human-filtered REDFM."""


class MultilingualJsonlSource(JsonlHubSource):
    """Streams a corpus laid out as one file per language and split.

    Args:
        repo_id: Dataset repository identifier.
        name: Short corpus identifier.
        languages: Languages to read, in the order they should be interleaved.
        template: Repository path template taking ``split`` and ``language``.
        available: Languages the corpus actually provides.
        cache_dir: Optional download cache directory.
        token: Optional Hub token.

    Raises:
        ValueError: If a requested language is not provided by the corpus.
    """

    def __init__(
        self,
        repo_id: str,
        name: str,
        languages: tuple[str, ...],
        template: str,
        available: tuple[str, ...],
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            name=name,
            provides_relations=True,
            cache_dir=cache_dir,
            token=token,
        )
        missing = [language for language in languages if language not in available]
        if missing:
            raise ValueError(
                f"{name} does not provide {', '.join(missing)}. "
                f"Available languages: {', '.join(available)}."
            )
        self._languages = languages
        self._template = template

    @property
    def languages(self) -> tuple[str, ...]:
        """Languages this reader will stream."""
        return self._languages

    def resolve_filename(self, split: str) -> str:
        """Return the repository path for a ``split:language`` key.

        Args:
            split: Either ``split`` or ``split:language``.

        Returns:
            Path of the file inside the dataset repository.
        """
        name, _, language = split.partition(":")
        return self._template.format(split=name, language=language or self._languages[0])

    def iter_records(self, split: str, limit: int | None = None) -> Iterator[dict]:
        """Stream records across every configured language.

        Languages are read round-robin rather than one after another so that a
        truncated pass still sees every language instead of exhausting the first
        one alphabetically.

        Args:
            split: Split name, without a language suffix.
            limit: Optional cap on the total number of records yielded.

        Yields:
            Records annotated with their language.
        """
        streams = [
            super(MultilingualJsonlSource, self).iter_records(f"{split}:{language}")
            for language in self._languages
        ]
        emitted = 0
        exhausted = set()
        while len(exhausted) < len(streams):
            for index, stream in enumerate(streams):
                if index in exhausted:
                    continue
                try:
                    record = next(stream)
                except StopIteration:
                    exhausted.add(index)
                    continue
                if limit is not None and emitted >= limit:
                    return
                record.setdefault("lan", self._languages[index])
                emitted += 1
                yield record


class SredfmSource(MultilingualJsonlSource):
    """Streams the automatically generated SREDFM corpus.

    Args:
        languages: Languages to read.
        cache_dir: Optional download cache directory.
        token: Optional Hub token.
    """

    def __init__(
        self,
        languages: tuple[str, ...],
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(
            repo_id="Babelscape/SREDFM",
            name="sredfm",
            languages=languages,
            template="data/{split}.{language}.jsonl",
            available=SREDFM_LANGUAGES,
            cache_dir=cache_dir,
            token=token,
        )


class RedfmSource(MultilingualJsonlSource):
    """Streams the human-filtered REDFM corpus.

    Args:
        languages: Languages to read. Languages outside REDFM's coverage are
            dropped with the caller's knowledge rather than raising, so a
            deployment configured for eight languages can still evaluate on the
            seven REDFM provides.
        cache_dir: Optional download cache directory.
        token: Optional Hub token.
    """

    def __init__(
        self,
        languages: tuple[str, ...],
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        covered = tuple(
            language for language in languages if language in REDFM_LANGUAGES
        )
        super().__init__(
            repo_id="Babelscape/REDFM",
            name="redfm",
            languages=covered or REDFM_LANGUAGES[:1],
            template="data/{split}.{language}.jsonl",
            available=REDFM_LANGUAGES,
            cache_dir=cache_dir,
            token=token,
        )
        self._requested = languages

    @property
    def uncovered_languages(self) -> tuple[str, ...]:
        """Requested languages REDFM cannot evaluate."""
        return tuple(
            language
            for language in self._requested
            if language not in REDFM_LANGUAGES
        )
