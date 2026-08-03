"""Splitting raw input text into words.

The training corpora arrive tokenised with punctuation separated from words.
Inference receives a plain string, so it has to reproduce that shape: feeding
differently segmented text would shift the input distribution for no reason.
"""

from __future__ import annotations

import re

WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class WordTokenizer:
    """Splits text into corpus-style word tokens.

    Punctuation becomes its own token, matching the corpus the model was trained
    on. This is deliberately not a linguistic tokenizer: it only has to agree
    with the training segmentation.
    """

    def split(self, text: str) -> list[str]:
        """Split a string into word tokens.

        Args:
            text: Raw input text.

        Returns:
            The word tokens, in order.
        """
        return WORD_PATTERN.findall(text)
