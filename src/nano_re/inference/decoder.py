"""Decoding BIO logits into entity mentions.

Training labels only the first sub-word of each word, so decoding reads the
logits at exactly those positions and ignores the rest.
"""

from __future__ import annotations

import torch

from ..schema import LabelSchema
from .results import PredictedMention


class BioSpanDecoder:
    """Turns token classification logits into typed mention spans.

    Args:
        schema: Label vocabularies used to map indices onto BIO tags.
    """

    def __init__(self, schema: LabelSchema) -> None:
        self._id_to_bio = schema.id_to_bio

    def decode(
        self,
        ner_logits: torch.Tensor,
        word_to_subwords: dict[int, list[int]],
        words: list[str],
    ) -> list[tuple[PredictedMention, str]]:
        """Extract mentions from the logits of a single document.

        A span ends when a word is tagged ``O``, tagged ``B-`` again, or tagged
        ``I-`` of a different type. An ``I-`` tag with no preceding ``B-`` opens
        a span rather than being discarded, which is the forgiving reading and
        recovers mentions the tagger started mid-way.

        Args:
            ner_logits: Token scores for one document, shape ``[S, L]``.
            word_to_subwords: Mapping from word index to sub-word positions.
            words: The tokenised input.

        Returns:
            Pairs of mention and entity type, in reading order.
        """
        tags = self._word_tags(ner_logits, word_to_subwords, len(words))
        mentions: list[tuple[PredictedMention, str]] = []
        start: int | None = None
        active_type = ""

        for index in range(len(words)):
            tag = tags.get(index, "O")
            prefix, _, entity_type = tag.partition("-")

            if prefix == "B" or (prefix == "I" and entity_type != active_type):
                if start is not None:
                    mentions.append(self._build(words, start, index, active_type))
                start = index
                active_type = entity_type
            elif prefix == "O":
                if start is not None:
                    mentions.append(self._build(words, start, index, active_type))
                start = None
                active_type = ""

        if start is not None:
            mentions.append(self._build(words, start, len(words), active_type))
        return mentions

    def _word_tags(
        self,
        ner_logits: torch.Tensor,
        word_to_subwords: dict[int, list[int]],
        num_words: int,
    ) -> dict[int, str]:
        """Read the predicted tag of every word that survived truncation.

        Args:
            ner_logits: Token scores for one document, shape ``[S, L]``.
            word_to_subwords: Mapping from word index to sub-word positions.
            num_words: Number of words in the input.

        Returns:
            Mapping from word index to BIO tag name.
        """
        predictions = ner_logits.argmax(dim=-1)
        tags: dict[int, str] = {}
        for index in range(num_words):
            positions = word_to_subwords.get(index)
            if not positions:
                continue
            tags[index] = self._id_to_bio[int(predictions[positions[0]])]
        return tags

    def _build(
        self, words: list[str], start: int, end: int, entity_type: str
    ) -> tuple[PredictedMention, str]:
        """Assemble a mention from a word range.

        Args:
            words: The tokenised input.
            start: Inclusive word index.
            end: Exclusive word index.
            entity_type: Coarse type of the mention.

        Returns:
            The mention paired with its type.
        """
        return (
            PredictedMention(
                text=" ".join(words[start:end]), start=start, end=end
            ),
            entity_type,
        )
