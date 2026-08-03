"""Deterministic extraction of structured identifiers.

Invoice numbers, tax identifiers and bank accounts have fixed shapes and
checksums, and appear in no NLP training corpus. They are matched by rule rather
than predicted by the model.
"""

from .extractor import PatternExtractor, PatternMatch
from .library import POLISH_BUSINESS_RULES
from .rules import (
    PatternRule,
    is_valid_iban,
    is_valid_nip,
    is_valid_pesel,
    is_valid_polish_account,
    is_valid_regon,
)

__all__ = [
    "POLISH_BUSINESS_RULES",
    "PatternExtractor",
    "PatternMatch",
    "PatternRule",
    "is_valid_iban",
    "is_valid_nip",
    "is_valid_pesel",
    "is_valid_polish_account",
    "is_valid_regon",
]
