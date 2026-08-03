"""Reader for the MultiNERD entity recognition corpus.

MultiNERD supplies pre-tokenised sentences with BIO tags and no relations. It is
here to give the token classification head far more multilingual supervision
than a relation corpus alone provides, and its lack of relations is declared
through :attr:`provides_relations` so the relation head is never trained on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .jsonl import JsonlHubSource

MULTINERD_LANGUAGES: tuple[str, ...] = (
    "de", "en", "es", "fr", "it", "nl", "pl", "pt", "ru", "zh",
)
"""Languages carried by MultiNERD."""


class MultiNerdSource(JsonlHubSource):
    """Streams MultiNERD, interleaving the configured languages.

    Args:
        languages: Languages to read. Languages outside MultiNERD's coverage are
            skipped so a deployment spanning more languages still gets the
            supervision MultiNERD can offer.
        cache_dir: Optional download cache directory.
        token: Optional Hub token.

    Raises:
        ValueError: If none of the requested languages are covered.
    """

    def __init__(
        self,
        languages: tuple[str, ...],
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(
            repo_id="Babelscape/multinerd",
            name="multinerd",
            provides_relations=False,
            cache_dir=cache_dir,
            token=token,
        )
        covered = tuple(
            language for language in languages if language in MULTINERD_LANGUAGES
        )
        if not covered:
            raise ValueError(
                "MultiNERD covers none of the requested languages. "
                f"Available: {', '.join(MULTINERD_LANGUAGES)}."
            )
        self._languages = covered
        self._requested = languages

    @property
    def languages(self) -> tuple[str, ...]:
        """Languages this reader will stream."""
        return self._languages

    @property
    def uncovered_languages(self) -> tuple[str, ...]:
        """Requested languages MultiNERD does not provide."""
        return tuple(
            language
            for language in self._requested
            if language not in MULTINERD_LANGUAGES
        )

    def resolve_filename(self, split: str) -> str:
        """Return the repository path for a ``split:language`` key.

        Args:
            split: Either ``split`` or ``split:language``.

        Returns:
            Path of the file inside the dataset repository.
        """
        name, _, language = split.partition(":")
        return f"{name}/{name}_{language or self._languages[0]}.jsonl"

    def iter_records(self, split: str, limit: int | None = None) -> Iterator[dict]:
        """Stream records across every configured language, round-robin.

        Args:
            split: Split name, without a language suffix.
            limit: Optional cap on the total number of records yielded.

        Yields:
            Records annotated with their language.
        """
        streams = [
            super(MultiNerdSource, self).iter_records(f"{split}:{language}")
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
                record.setdefault("lang", self._languages[index])
                emitted += 1
                yield record
