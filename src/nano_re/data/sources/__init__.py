"""Corpus readers.

Each reader streams raw records from one corpus and declares whether that corpus
supervises the relation head.
"""

from .base import DocumentSource
from .jsonl import JsonlHubSource
from .multinerd import MULTINERD_LANGUAGES, MultiNerdSource
from .redfm import (
    REDFM_LANGUAGES,
    SREDFM_LANGUAGES,
    MultilingualJsonlSource,
    RedfmSource,
    SredfmSource,
)

__all__ = [
    "DocumentSource",
    "JsonlHubSource",
    "MULTINERD_LANGUAGES",
    "MultiNerdSource",
    "MultilingualJsonlSource",
    "REDFM_LANGUAGES",
    "RedfmSource",
    "SREDFM_LANGUAGES",
    "SredfmSource",
]
