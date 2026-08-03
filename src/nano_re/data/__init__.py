"""Data acquisition, parsing, encoding and batching."""

from .collator import MultiTaskBatch, MultiTaskCollator
from .document import Document, Entity, Mention, RelationTriple
from .encoder import DocumentEncoder, EncodedDocument
from .module import CorpusBundle, DataModule
from .multi_corpus import CorpusSpec, CorpusStatistics, MultiCorpusDataset
from .parsers import MultiNerdParser, SredfmParser
from .sources import DocumentSource, MultiNerdSource, RedfmSource, SredfmSource

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
    "MultiNerdParser",
    "MultiNerdSource",
    "MultiTaskBatch",
    "MultiTaskCollator",
    "RedfmSource",
    "RelationTriple",
    "SredfmParser",
    "SredfmSource",
]
