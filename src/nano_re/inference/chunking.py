"""Windowing long documents and merging the results back together.

The encoder accepts 512 sub-word tokens. How much text that is depends entirely
on the language and the content: measured, English prose runs 1.48 sub-words per
word, Polish prose 1.77, and a Polish invoice 3.13, because identifiers shatter
into many pieces. Sizing windows by word count would therefore overflow on
exactly the documents this model is meant for.

Windows are instead packed against a measured sub-word budget, and they overlap
so that an entity or a relation spanning a boundary is still seen whole by at
least one window.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from .results import (
    ExtractionResult,
    PredictedEntity,
    PredictedMention,
    PredictedRelation,
)


@dataclass(frozen=True)
class Window:
    """One slice of a document.

    Attributes:
        start: Inclusive word index in the full document.
        end: Exclusive word index in the full document.
    """

    start: int
    end: int

    @property
    def size(self) -> int:
        """Number of words in the window."""
        return self.end - self.start


class TextChunker:
    """Packs words into overlapping windows that fit the encoder.

    Args:
        tokenizer: Tokenizer whose sub-word counts define the budget.
        max_sequence_length: Encoder window in sub-word tokens.
        overlap: Fraction of each window repeated in the next one.
        reserved_tokens: Positions kept free for the special tokens the
            tokenizer adds around a sequence.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_sequence_length: int = 512,
        overlap: float = 0.25,
        reserved_tokens: int = 4,
    ) -> None:
        self._tokenizer = tokenizer
        self._budget = max(1, max_sequence_length - reserved_tokens)
        self._overlap = min(max(overlap, 0.0), 0.9)
        self._lengths: dict[str, int] = {}

    def subword_length(self, word: str) -> int:
        """Return how many sub-words a word costs, memoised.

        Args:
            word: A single word.

        Returns:
            The sub-word count, at least one.
        """
        if word not in self._lengths:
            encoded = self._tokenizer(
                [word], is_split_into_words=True, add_special_tokens=False
            )["input_ids"]
            self._lengths[word] = max(1, len(encoded))
        return self._lengths[word]

    def split(self, words: list[str]) -> list[Window]:
        """Divide a document into windows that each fit the budget.

        Args:
            words: The tokenised document.

        Returns:
            Windows covering every word, in order. A document that already fits
            yields a single window.
        """
        if not words:
            return []

        windows: list[Window] = []
        start = 0
        while start < len(words):
            used = 0
            end = start
            while end < len(words):
                cost = self.subword_length(words[end])
                if used + cost > self._budget and end > start:
                    break
                used += cost
                end += 1
            windows.append(Window(start=start, end=end))
            if end >= len(words):
                break
            step = max(1, int((end - start) * (1.0 - self._overlap)))
            start += step
        return windows


class ResultMerger:
    """Combines per-window extractions into one document-level result.

    Entities are merged across windows by normalised surface form and type, so
    the same organisation seen in two windows becomes one entity with two
    mentions rather than two entities. Relations are deduplicated on their
    endpoints and predicate, keeping the highest confidence observed: a pair the
    model saw twice should be reported once, at its strongest reading.
    """

    def merge(
        self,
        results: list[tuple[Window, ExtractionResult]],
        words: tuple[str, ...],
    ) -> ExtractionResult:
        """Merge the results of every window.

        Args:
            results: Window and its extraction, in document order.
            words: The full tokenised document.

        Returns:
            One result whose mention offsets are document-global.
        """
        if not results:
            return ExtractionResult(words=words, entities=(), relations=())

        keys: dict[tuple[str, str], int] = {}
        mentions: list[list[PredictedMention]] = []
        types: list[str] = []
        names: list[str] = []
        local_to_global: list[dict[int, int]] = []

        for window, result in results:
            mapping: dict[int, int] = {}
            for entity in result.entities:
                key = (entity.entity_type, _normalise(entity.name))
                if key not in keys:
                    keys[key] = len(mentions)
                    mentions.append([])
                    types.append(entity.entity_type)
                    names.append(entity.name)
                index = keys[key]
                mapping[entity.index] = index
                if len(entity.name) > len(names[index]):
                    names[index] = entity.name
                for mention in entity.mentions:
                    shifted = PredictedMention(
                        text=mention.text,
                        start=mention.start + window.start,
                        end=mention.end + window.start,
                    )
                    if shifted not in mentions[index]:
                        mentions[index].append(shifted)
            local_to_global.append(mapping)

        entities = tuple(
            PredictedEntity(
                index=index,
                name=names[index],
                entity_type=types[index],
                mentions=tuple(
                    sorted(mentions[index], key=lambda item: item.start)
                ),
            )
            for index in range(len(mentions))
        )

        best: dict[tuple[int, int, str], PredictedRelation] = {}
        for (window, result), mapping in zip(results, local_to_global):
            for relation in result.relations:
                head = mapping.get(relation.head)
                tail = mapping.get(relation.tail)
                if head is None or tail is None or head == tail:
                    continue
                key = (head, tail, relation.relation)
                candidate = PredictedRelation(
                    head=head,
                    tail=tail,
                    relation=relation.relation,
                    label=relation.label,
                    confidence=relation.confidence,
                )
                if key not in best or candidate.confidence > best[key].confidence:
                    best[key] = candidate

        relations = tuple(
            sorted(best.values(), key=lambda item: item.confidence, reverse=True)
        )
        return ExtractionResult(
            words=words, entities=entities, relations=relations, truncated_words=0
        )


def _normalise(text: str) -> str:
    """Reduce a surface form to a merge key.

    Args:
        text: Raw surface form.

    Returns:
        Lower-cased text with whitespace collapsed.
    """
    return " ".join(text.lower().split())
