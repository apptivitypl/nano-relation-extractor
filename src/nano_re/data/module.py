"""Composition root for the data pipeline.

The module builds corpora, the label schema and the loaders, and is the only
data component the notebook or CLI needs to touch.

The label schema is derived from the corpora rather than declared up front. Which
relations exist depends on which languages are in scope, so the inventory is
counted during a first pass and then frozen into the bundle, where inference and
the model card read the same file the heads were sized from.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..config import DataConfig, ModelConfig
from ..schema import LabelSchema, RelationInventory
from .collator import MultiTaskCollator
from .encoder import DocumentEncoder
from .multi_corpus import CorpusSpec, CorpusStatistics, MultiCorpusDataset
from .parsers import DocRedParser, KpwrParser, SredfmParser
from .sources import KpwrSource, ReDocredSource, RedfmSource, SredfmSource


@dataclass(frozen=True)
class CorpusBundle:
    """A dataset together with the counters describing how it was built.

    Attributes:
        dataset: The interleaved, lazily encoding dataset.
        statistics: Per-corpus counters.
    """

    dataset: MultiCorpusDataset
    statistics: tuple[CorpusStatistics, ...]

    def describe(self) -> str:
        """Return a human readable summary of every contributing corpus."""
        lines = [f"Stream: {len(self.dataset)} documents"]
        lines.extend(item.describe() for item in self.statistics)
        return "\n".join(lines)


class DataModule:
    """Builds tokenizer, schema and data loaders from configuration.

    Args:
        data_config: Corpus, language and encoding settings.
        model_config: Provides the backbone name used to select the tokenizer.
    """

    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        self._data_config = data_config
        self._model_config = model_config
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._schema: LabelSchema | None = None
        self._encoder: DocumentEncoder | None = None
        self._inventory = RelationInventory()

    @property
    def languages(self) -> tuple[str, ...]:
        """Languages this module reads."""
        return self._data_config.languages

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizer matching the configured backbone, loaded on first use."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_config.backbone_name
            )
        return self._tokenizer

    @property
    def inventory(self) -> RelationInventory:
        """Relation counts accumulated while parsing."""
        return self._inventory

    @property
    def schema(self) -> LabelSchema:
        """Label schema, built from the corpora on first access.

        Raises:
            RuntimeError: If no relation has been seen yet, which means the
                corpora were never read.
        """
        if self._schema is None:
            if not len(self._inventory):
                raise RuntimeError(
                    "The relation inventory is empty. Build a training corpus "
                    "before requesting the schema."
                )
            self._schema = self._inventory.to_schema(
                languages=self._data_config.languages,
                min_count=self._data_config.min_relation_count,
                max_relations=self._data_config.max_relations,
            )
        return self._schema

    def set_schema(self, schema: LabelSchema) -> None:
        """Adopt an existing schema instead of deriving one.

        Args:
            schema: Schema loaded from a bundle, so that a later stage sizes its
                heads exactly as the trained checkpoint did.
        """
        self._schema = schema
        self._encoder = None

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

    def build_relation_source(self, gold: bool = False):
        """Create the reader for the relation corpus.

        Args:
            gold: Whether to build the human-filtered evaluation corpus instead
                of the automatically generated training corpus.

        Returns:
            The configured reader.
        """
        if gold:
            return RedfmSource(languages=self._data_config.languages)
        return SredfmSource(languages=self._data_config.languages)

    def build_entity_source(self):
        """Create the reader for the entity-only corpus.

        Returns:
            The configured reader.
        """
        return KpwrSource(languages=self._data_config.languages)

    def build_english_relation_source(self):
        """Create the reader for the English document-level relation corpus.

        Returns:
            The configured reader.
        """
        return ReDocredSource(languages=self._data_config.languages)

    def build_corpus(
        self, split: str, training: bool, gold: bool = False
    ) -> CorpusBundle:
        """Build an interleaved dataset for one split.

        Parsing happens here, which is also when the relation inventory grows.
        A schema derived afterwards therefore reflects exactly the data the
        model will be trained on.

        Args:
            split: Split name passed to every reader.
            training: When ``True`` negative pairs are subsampled and the entity
                corpus is included. Evaluation uses the relation corpus alone,
                since a corpus without relations cannot score the relation head.
            gold: Whether to read the human-filtered relation corpus.

        Returns:
            The dataset and its per-corpus counters.
        """
        limit = self._data_config.limit
        specs = [
            CorpusSpec(
                name=self._data_config.relation_corpus,
                source=self.build_relation_source(gold=gold),
                parser=SredfmParser(inventory=self._inventory),
                split=split,
                weight=self._data_config.relation_weight,
                limit=limit,
            )
        ]
        if training and self._data_config.entity_weight > 0:
            specs.append(
                CorpusSpec(
                    name=self._data_config.entity_corpus,
                    source=self.build_entity_source(),
                    parser=KpwrParser(),
                    split=split,
                    weight=self._data_config.entity_weight,
                    limit=limit,
                )
            )
        if training and self._data_config.english_relation_weight > 0:
            specs.append(
                CorpusSpec(
                    name=self._data_config.english_relation_corpus,
                    source=self.build_english_relation_source(),
                    parser=DocRedParser(inventory=self._inventory),
                    split=split,
                    weight=self._data_config.english_relation_weight,
                    limit=limit,
                )
            )

        dataset = MultiCorpusDataset(
            specs=specs,
            encoder_factory=lambda: self.encoder,
            sample_negatives=training,
            cache=self._should_cache(specs, limit),
        )
        return CorpusBundle(
            dataset=dataset, statistics=tuple(dataset.statistics)
        )

    def build_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        """Wrap a dataset in a data loader.

        Args:
            dataset: Dataset yielding encoded documents or ``None``.
            batch_size: Documents per batch.
            shuffle: Whether to shuffle between epochs.

        Returns:
            A configured :class:`torch.utils.data.DataLoader`.

        Raises:
            ValueError: If the tokenizer defines no padding token.
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer does not define a padding token.")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._data_config.num_workers,
            collate_fn=MultiTaskCollator(pad_token_id=pad_token_id),
        )

    def _should_cache(self, specs: list[CorpusSpec], limit: int | None) -> bool:
        """Decide whether encoded tensors may be retained between epochs.

        Args:
            specs: Corpora contributing to the stream.
            limit: Per-corpus record cap, when one is configured.

        Returns:
            ``True`` when the stream is small enough to cache.
        """
        if limit is None:
            return False
        return limit * len(specs) <= self._data_config.max_cached_documents
