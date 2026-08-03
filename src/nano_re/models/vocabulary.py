"""Trimming the embedding table to the languages actually in scope.

The multilingual encoder carries a 250k token vocabulary covering more than a
hundred languages, and that table is 96 of the model's 107.7M parameters. A
deployment that handles eight European languages pays for the other hundred in
memory and file size and gets nothing back.

Trimming keeps the tokens the corpora actually use and rebuilds the embedding
matrix around them. The tokenizer is deliberately left untouched: rewriting a
SentencePiece vocabulary is fragile, and it is not necessary. Instead the model
carries a lookup table from original token identifier to compacted row, applied
before the embedding lookup. The table costs a few megabytes against the tens
saved, exports to ONNX as an ordinary gather, and means every tokenizer file in
the bundle stays exactly what the encoder was pretrained with.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TrimReport:
    """Outcome of a vocabulary trim.

    Attributes:
        original_size: Token count before trimming.
        trimmed_size: Token count kept.
        coverage: Fraction of observed token occurrences the kept set covers.
        documents_sampled: Documents inspected while counting.
        parameters_removed: Embedding parameters eliminated.
    """

    original_size: int
    trimmed_size: int
    coverage: float
    documents_sampled: int
    parameters_removed: int

    @property
    def reduction(self) -> float:
        """Fraction of the vocabulary removed."""
        if not self.original_size:
            return 0.0
        return 1.0 - (self.trimmed_size / self.original_size)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the report."""
        return {
            "original_size": self.original_size,
            "trimmed_size": self.trimmed_size,
            "coverage": self.coverage,
            "documents_sampled": self.documents_sampled,
            "parameters_removed": self.parameters_removed,
            "reduction": self.reduction,
        }

    def describe(self) -> str:
        """Return a one-line human readable summary."""
        return (
            f"Slownik: {self.original_size} -> {self.trimmed_size} tokenow "
            f"({self.reduction:.1%} mniej), pokrycie {self.coverage:.4f} "
            f"na {self.documents_sampled} dokumentach, "
            f"{self.parameters_removed / 1e6:.1f}M parametrow usunietych."
        )


class VocabularyTrimmer:
    """Counts token usage over a corpus and compacts the embedding table.

    Args:
        tokenizer: Tokenizer whose identifiers are being counted.
        target_coverage: Fraction of token occurrences the kept set must cover.
            The tail below this threshold is dropped and folded into the unknown
            token.
        min_vocab_size: Floor on the kept vocabulary, guarding against a tiny
            sample producing a uselessly small table.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        target_coverage: float = 0.9999,
        min_vocab_size: int = 8000,
    ) -> None:
        self._tokenizer = tokenizer
        self._target_coverage = target_coverage
        self._min_vocab_size = min_vocab_size
        self._counts: Counter[int] = Counter()
        self._documents = 0

    @property
    def documents_sampled(self) -> int:
        """Number of documents counted so far."""
        return self._documents

    def observe_words(self, words: list[str]) -> None:
        """Count the tokens a tokenised document uses.

        Args:
            words: Tokenised document.
        """
        if not words:
            return
        encoded = self._tokenizer(
            words, is_split_into_words=True, add_special_tokens=False
        )["input_ids"]
        self._counts.update(encoded)
        self._documents += 1

    def observe_documents(self, documents) -> None:
        """Count tokens across an iterable of parsed documents.

        Args:
            documents: Iterable of objects exposing a ``words`` sequence.
        """
        for document in documents:
            self.observe_words(list(document.words))

    def kept_token_ids(self) -> list[int]:
        """Return the token identifiers to retain, in ascending order.

        Special tokens are always retained regardless of frequency: dropping a
        padding or unknown token would break the tokenizer contract rather than
        merely lose coverage.

        Returns:
            Sorted original token identifiers.
        """
        required = {
            identifier
            for identifier in self._tokenizer.all_special_ids
            if identifier is not None
        }
        total = sum(self._counts.values())
        if not total:
            return sorted(required)

        kept = set(required)
        accumulated = 0
        for identifier, count in self._counts.most_common():
            if (
                accumulated / total >= self._target_coverage
                and len(kept) >= self._min_vocab_size
            ):
                break
            kept.add(identifier)
            accumulated += count
        return sorted(kept)

    def coverage_of(self, kept: list[int]) -> float:
        """Return the fraction of observed occurrences a token set covers.

        Args:
            kept: Token identifiers retained.

        Returns:
            The coverage fraction, ``1.0`` when nothing was observed.
        """
        total = sum(self._counts.values())
        if not total:
            return 1.0
        retained = set(kept)
        return sum(
            count for identifier, count in self._counts.items() if identifier in retained
        ) / total

    def build_remap(self, kept: list[int], original_size: int) -> torch.Tensor:
        """Build the original-to-compacted identifier lookup table.

        Args:
            kept: Retained token identifiers, ascending.
            original_size: Size of the original vocabulary.

        Returns:
            A ``[original_size]`` long tensor. Dropped identifiers point at the
            compacted row of the unknown token.
        """
        unknown = self._tokenizer.unk_token_id
        fallback = kept.index(unknown) if unknown in kept else 0
        remap = torch.full((original_size,), fallback, dtype=torch.long)
        for compact, original in enumerate(kept):
            if 0 <= original < original_size:
                remap[original] = compact
        return remap

    def trim(self, backbone) -> TrimReport:
        """Compact a backbone's embedding table in place.

        Args:
            backbone: The encoder wrapper to modify.

        Returns:
            A report describing what was removed.

        Raises:
            RuntimeError: If no tokens were observed, which means the corpora
                were never sampled and trimming would discard everything.
        """
        if not self._counts:
            raise RuntimeError(
                "No tokens were observed. Sample the corpora before trimming."
            )

        embeddings = backbone.encoder.get_input_embeddings()
        original_size, hidden_size = embeddings.weight.shape
        kept = self.kept_token_ids()
        coverage = self.coverage_of(kept)

        index = torch.tensor(kept, dtype=torch.long)
        compacted = torch.nn.Embedding(len(kept), hidden_size)
        with torch.no_grad():
            compacted.weight.copy_(embeddings.weight.index_select(0, index))
        if embeddings.padding_idx is not None:
            padding = self._tokenizer.pad_token_id
            if padding in kept:
                compacted.padding_idx = kept.index(padding)

        backbone.encoder.set_input_embeddings(compacted)
        backbone.encoder.config.vocab_size = len(kept)
        backbone.attach_token_remap(self.build_remap(kept, original_size))

        return TrimReport(
            original_size=original_size,
            trimmed_size=len(kept),
            coverage=coverage,
            documents_sampled=self._documents,
            parameters_removed=(original_size - len(kept)) * hidden_size,
        )
