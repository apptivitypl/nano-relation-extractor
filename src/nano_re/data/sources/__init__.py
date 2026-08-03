"""Corpus readers.

Each reader streams raw records from one corpus and declares whether that corpus
supervises the relation head. Every corpus here carries a licence permitting
commercial use; that is a hard requirement, not a preference, because a model
trained on non-commercial data would be one nobody could deploy.
"""

from .base import DocumentSource
from .jsonl import JsonlHubSource
from .kpwr import KpwrSource
from .redfm import (
    REDFM_LANGUAGES,
    SREDFM_LANGUAGES,
    MultilingualJsonlSource,
    RedfmSource,
    SredfmSource,
)
from .redocred import ReDocredSource

__all__ = [
    "DocumentSource",
    "JsonlHubSource",
    "KpwrSource",
    "MultilingualJsonlSource",
    "REDFM_LANGUAGES",
    "ReDocredSource",
    "RedfmSource",
    "SREDFM_LANGUAGES",
    "SredfmSource",
]
