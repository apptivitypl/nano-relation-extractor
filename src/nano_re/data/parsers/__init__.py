"""Corpus specific record parsers.

Each parser turns one corpus's raw records into :class:`Document` objects, so
everything downstream is corpus agnostic.
"""

from .docred import DocRedParser
from .kpwr import KpwrParser, canonical_kpwr_type
from .sredfm import SredfmParser, WordSpan

__all__ = [
    "DocRedParser",
    "KpwrParser",
    "SredfmParser",
    "WordSpan",
    "canonical_kpwr_type",
]
