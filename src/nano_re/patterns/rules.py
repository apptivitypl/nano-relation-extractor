"""Rule definitions for deterministic identifier extraction.

Structured identifiers are the part of a business document a neural tagger is
worst at and a regular expression is best at. A NIP has a fixed shape and a
checksum; a model trained on encyclopaedic prose has never seen one. Matching
them by rule is not a fallback, it is the correct tool.

A rule may carry a validator. Without one, ``123-456-78-90`` matches the NIP
shape and is reported; with the checksum validator it is rejected, which is what
separates a usable extractor from one that floods the output with noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PatternRule:
    """One deterministic extraction rule.

    Attributes:
        name: Identifier of the rule, used in diagnostics.
        entity_type: Type assigned to matches, reported alongside model output.
        pattern: Compiled expression. Group ``value`` selects the reported span
            when present, otherwise the whole match is used.
        validator: Optional check applied to the normalised value. A rule whose
            validator rejects a match reports nothing.
        normaliser: Optional transform producing the canonical form recorded for
            the match, such as stripping separators from a NIP.
        priority: Higher values win when two rules match overlapping spans.
    """

    name: str
    entity_type: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None
    normaliser: Callable[[str], str] | None = None
    priority: int = 0

    def normalise(self, raw: str) -> str:
        """Return the canonical form of a matched string.

        Args:
            raw: Text as it appears in the document.

        Returns:
            The normalised value, or the input when the rule defines no
            normaliser.
        """
        return self.normaliser(raw) if self.normaliser else raw

    def accepts(self, value: str) -> bool:
        """Test whether a normalised value passes this rule's validator.

        Args:
            value: Normalised candidate value.

        Returns:
            ``True`` when the rule has no validator or the validator accepts.
        """
        return self.validator is None or self.validator(value)


def digits_only(text: str) -> str:
    """Strip everything except digits.

    Args:
        text: Raw matched text.

    Returns:
        The digits of the input, in order.
    """
    return re.sub(r"\D", "", text)


def alphanumeric_upper(text: str) -> str:
    """Strip separators and upper-case the remainder.

    Args:
        text: Raw matched text.

    Returns:
        The input without spaces or punctuation, upper-cased.
    """
    return re.sub(r"[^0-9A-Za-z]", "", text).upper()


def _weighted_modulo_check(value: str, weights: tuple[int, ...], modulo: int) -> bool:
    """Verify a digit string against a weighted checksum.

    Args:
        value: Digit string including its trailing check digit.
        weights: Weight applied to each digit before the check digit.
        modulo: Modulus of the checksum.

    Returns:
        ``True`` when the computed check digit matches the trailing digit.
    """
    if len(value) != len(weights) + 1 or not value.isdigit():
        return False
    total = sum(int(digit) * weight for digit, weight in zip(value, weights))
    check = total % modulo
    return check != 10 and check == int(value[-1])


def is_valid_nip(value: str) -> bool:
    """Validate a Polish tax identification number.

    Args:
        value: Ten digit NIP without separators.

    Returns:
        ``True`` when the checksum is correct.
    """
    return _weighted_modulo_check(value, (6, 5, 7, 2, 3, 4, 5, 6, 7), 11)


def is_valid_regon(value: str) -> bool:
    """Validate a Polish business registry number.

    Both the nine and fourteen digit forms are accepted, each with its own
    weight vector.

    Args:
        value: Nine or fourteen digit REGON without separators.

    Returns:
        ``True`` when the checksum is correct.
    """
    if len(value) == 9:
        return _weighted_modulo_check(value, (8, 9, 2, 3, 4, 5, 6, 7), 11)
    if len(value) == 14:
        if not _weighted_modulo_check(value[:9], (8, 9, 2, 3, 4, 5, 6, 7), 11):
            return False
        return _weighted_modulo_check(
            value, (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8), 11
        )
    return False


def is_valid_pesel(value: str) -> bool:
    """Validate a Polish personal identification number.

    Args:
        value: Eleven digit PESEL.

    Returns:
        ``True`` when the checksum is correct.
    """
    if len(value) != 11 or not value.isdigit():
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    total = sum(int(digit) * weight for digit, weight in zip(value, weights))
    return (10 - total % 10) % 10 == int(value[-1])


def is_valid_iban(value: str) -> bool:
    """Validate an IBAN using the ISO 13616 modulo 97 check.

    Args:
        value: IBAN without separators, including the country prefix.

    Returns:
        ``True`` when the checksum is correct.
    """
    if len(value) < 15 or len(value) > 34 or not value[:2].isalpha():
        return False
    rearranged = value[4:] + value[:4]
    digits = "".join(
        str(int(character, 36)) if character.isalpha() else character
        for character in rearranged
    )
    if not digits.isdigit():
        return False
    return int(digits) % 97 == 1


def is_valid_polish_account(value: str) -> bool:
    """Validate a bare Polish bank account number.

    A Polish account written without its ``PL`` prefix is still an IBAN body, so
    it is validated by prefixing the country code before the modulo 97 check.

    Args:
        value: Twenty-six digit account number.

    Returns:
        ``True`` when the checksum is correct.
    """
    if len(value) != 26 or not value.isdigit():
        return False
    return is_valid_iban("PL" + value)
