"""Composition root for the data pipeline.

The module wires source, parser, encoder and collator together and is the only
data component the notebook or CLI needs to touch.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..config import DataConfig, ModelConfig
from ..schema import LabelSchema
from .collator import MultiTaskCollator
from .document import Document
from .encoder import DocumentEncoder, EncodedDocument
from .parser import DocRedParser
from .source import DocREDHubSource, DocumentSource


@dataclass(frozen=True)
class SplitStatistics:
    """Counters describing what a split contributed after encoding.

    Attributes:
        split: Split name.
        raw_documents: Records downloaded from the corpus.
        encoded_documents: Records that produced usable tensors.
        gold_triples: Gold relation triples retained after truncation.
        dropped_triples: Gold triples lost because an endpoint was truncated.
    """

    split: str
    raw_documents: int
    encoded_documents: int
    gold_triples: int
    dropped_triples: int

    @property
    def recall_ceiling(self) -> float:
        """Maximum achievable relation recall given truncation losses."""
        total = self.gold_triples + self.dropped_triples
        return self.gold_triples / total if total else 1.0


class DataModule:
    """Builds tokenizer, schema and data loaders from configuration.

    Args:
        data_config: Dataset and encoding settings.
        model_config: Provides the backbone name used to select the tokenizer.
        source: Corpus reader. Defaults to a DocRED Hub reader.
    """

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        source: DocumentSource | None = None,
    ) -> None:
        self._data_config = data_config
        self._model_config = model_config
        self._source = source or DocREDHubSource(
            repo_id=data_config.dataset_repo_id,
            cache_dir=data_config.cache_dir,
        )
        self._parser = DocRedParser()
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._schema: LabelSchema | None = None
        self._encoder: DocumentEncoder | None = None
        self._statistics: dict[str, SplitStatistics] = {}

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizer matching the configured backbone, loaded on first use."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_config.backbone_name
            )
        return self._tokenizer

    @property
    def schema(self) -> LabelSchema:
        """Label vocabularies derived from the corpus relation inventory."""
        if self._schema is None:
            self._schema = LabelSchema.from_relation_info(
                self._source.load_relation_info()
            )
        return self._schema

    @property
    def encoder(self) -> DocumentEncoder:
        """Document encoder configured from the data settings."""
        if self._encoder is None:
            self._encoder = DocumentEncoder(
                tokenizer=self.tokenizer,
                schema=self.schema,
                max_sequence_length=self._data_config.max_sequence_length,
                max_negative_pairs=self._data_config.max_negative_pairs,
            )
        return self._encoder

    @property
    def statistics(self) -> dict[str, SplitStatistics]:
        """Per-split counters gathered during the most recent encoding."""
        return dict(self._statistics)

    def load_documents(self, split: str) -> list[Document]:
        """Download and parse a split into documents.

        Args:
            split: Split name understood by the configured source.

        Returns:
            The parsed documents.
        """
        raw = self._source.load_split(split, limit=self._data_config.limit)
        return self._parser.parse_all(raw)

    def encode_split(self, split: str, training: bool) -> list[EncodedDocument]:
        """Download, parse and encode a split, recording statistics.

        Args:
            split: Split name understood by the configured source.
            training: When ``True`` negative pairs are subsampled.

        Returns:
            The encoded documents.
        """
        documents = self.load_documents(split)
        encoded = self.encoder.encode_all(documents, sample_negatives=training)
        gold = sum(
            int(item.relation_labels[:, 1:].sum().item()) for item in encoded
        )
        self._statistics[split] = SplitStatistics(
            split=split,
            raw_documents=len(documents),
            encoded_documents=len(encoded),
            gold_triples=gold,
            dropped_triples=sum(item.dropped_relations for item in encoded),
        )
        return encoded

    def build_loader(
        self, documents: list[EncodedDocument], batch_size: int, shuffle: bool
    ) -> DataLoader:
        """Wrap encoded documents in a data loader.

        Args:
            documents: Encoded documents to iterate.
            batch_size: Documents per batch.
            shuffle: Whether to shuffle between epochs.

        Returns:
            A configured :class:`torch.utils.data.DataLoader`.
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer does not define a padding token.")
        return DataLoader(
            documents,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._data_config.num_workers,
            collate_fn=MultiTaskCollator(pad_token_id=pad_token_id),
        )

    def train_loader(self) -> DataLoader:
        """Build the training loader for the configured training split."""
        encoded = self.encode_split(self._data_config.train_split, training=True)
        return self.build_loader(
            encoded, self._data_config.train_batch_size, shuffle=True
        )

    def eval_loader(self) -> DataLoader:
        """Build the evaluation loader for the configured evaluation split."""
        encoded = self.encode_split(self._data_config.eval_split, training=False)
        return self.build_loader(
            encoded, self._data_config.eval_batch_size, shuffle=False
        )
