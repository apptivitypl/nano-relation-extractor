"""Data acquisition, parsing, encoding and batching."""

from .collator import MultiTaskBatch, MultiTaskCollator
from .document import Document, Entity, Mention, RelationTriple
from .encoder import DocumentEncoder, EncodedDocument
from .module import CorpusBundle, DataModule
from .multi_corpus import CorpusSpec, CorpusStatistics, MultiCorpusDataset
from .parsers import DocRedParser, KpwrParser, SredfmParser
from .sources import (
    DocumentSource,
    KpwrSource,
    ReDocredSource,
    RedfmSource,
    SredfmSource,
)

__all__ = [
    "CorpusBundle",
    "CorpusSpec",
    "CorpusStatistics",
    "DataModule",
    "Document",
    "DocumentEncoder",
    "DocumentSource",
    "EncodedDocument",
    "Entity",
    "Mention",
    "MultiCorpusDataset",
    "DocRedParser",
    "KpwrParser",
    "KpwrSource",
    "ReDocredSource",
    "MultiTaskBatch",
    "MultiTaskCollator",
    "RedfmSource",
    "RelationTriple",
    "SredfmParser",
    "SredfmSource",
]
