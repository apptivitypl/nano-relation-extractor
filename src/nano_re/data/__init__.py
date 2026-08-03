"""Data acquisition, parsing, encoding and batching."""

from .collator import MultiTaskBatch, MultiTaskCollator
from .document import Document, Entity, Mention, RelationTriple
from .encoder import DocumentEncoder, EncodedDocument
from .module import DataModule, SplitStatistics
from .parser import DocRedParser
from .source import DocREDHubSource, DocumentSource

__all__ = [
    "DataModule",
    "DocREDHubSource",
    "DocRedParser",
    "Document",
    "DocumentEncoder",
    "DocumentSource",
    "EncodedDocument",
    "Entity",
    "Mention",
    "MultiTaskBatch",
    "MultiTaskCollator",
    "RelationTriple",
    "SplitStatistics",
]
