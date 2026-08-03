"""Corpus specific record parsers.

Each parser turns one corpus's raw records into :class:`Document` objects, so
everything downstream is corpus agnostic.
"""

from .multinerd import MultiNerdParser
from .sredfm import SredfmParser, WordSpan

__all__ = ["MultiNerdParser", "SredfmParser", "WordSpan"]
