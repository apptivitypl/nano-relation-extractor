"""Immutable configuration objects for the whole pipeline.

Every stage of the pipeline receives a configuration object instead of reading
environment variables or literals on its own. Only this module knows how to
translate environment variables into typed settings, which keeps credential
handling and defaulting in a single auditable place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _env_str(name: str, default: str) -> str:
    """Return the environment value for ``name`` or ``default`` when unset.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or empty.

    Returns:
        The resolved string value.
    """
    value = os.getenv(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    """Return an integer environment value or ``default`` when unset.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or not an integer.

    Returns:
        The resolved integer value.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Return a float environment value or ``default`` when unset.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or not a float.

    Returns:
        The resolved float value.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment value or ``default`` when unset.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        ``True`` for ``1``, ``true``, ``yes`` and ``on`` (case-insensitive).
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DataConfig:
    """Settings describing dataset retrieval and encoding.

    Attributes:
        languages: Languages in scope. Determines which corpus shards are read
            and, through the vocabulary trimmer, how large the model ends up.
        relation_corpus: Corpus supervising both heads.
        entity_corpus: Corpus supervising the token head only. It exists because
            the relation corpus alone provides far less entity supervision than
            a multilingual tagger needs.
        english_relation_corpus: Human-annotated document-level relation corpus.
            English only, and included because it is the strongest relation
            supervision available anywhere and carries a permissive licence.
        english_relation_weight: Sampling weight of that corpus.
        gold_eval_corpus: Human-filtered corpus used for evaluation. It covers
            fewer languages than training does; the uncovered ones are reported
            rather than silently scored against silver data.
        relation_weight: Sampling weight of the relation corpus when
            interleaving.
        entity_weight: Sampling weight of the entity corpus.
        min_relation_count: Smallest number of occurrences for a relation to
            enter the label schema.
        max_relations: Optional cap on the relation inventory, keeping the most
            frequent. ``None`` keeps every relation meeting the minimum.
        train_split: Split used for training.
        eval_split: Split used for evaluation.
        max_sequence_length: Maximum number of sub-word tokens per document.
        max_negative_pairs: Number of sampled negative entity pairs per training
            document. Evaluation always scores every candidate pair.
        train_batch_size: Documents per training batch. Zero selects the
            measured default for the detected device.
        eval_batch_size: Documents per evaluation batch. Zero selects the
            measured default for the detected device.
        num_workers: Loader worker processes. Negative selects the measured
            default for the detected device.
        limit: Optional cap on the number of documents per split, used for smoke
            tests. ``None`` means no cap.
        max_cached_documents: Largest split kept as encoded tensors between
            epochs. Encoding costs roughly 68 KB per document, so the distantly
            supervised split would hold about 7 GB; above this threshold
            documents are re-encoded each epoch to bound memory instead.
        cache_dir: Directory for downloaded dataset files.
    """

    languages: tuple[str, ...] = ("pl", "en", "de", "fr", "es", "it", "nl", "pt")
    relation_corpus: str = "sredfm"
    entity_corpus: str = "kpwr"
    english_relation_corpus: str = "redocred"
    gold_eval_corpus: str = "redfm"
    relation_weight: float = 4.0
    entity_weight: float = 1.0
    english_relation_weight: float = 1.0
    min_relation_count: int = 1
    max_relations: int | None = None
    train_split: str = "train"
    eval_split: str = "dev"
    max_sequence_length: int = 512
    max_negative_pairs: int = 96
    train_batch_size: int = 0
    eval_batch_size: int = 0
    num_workers: int = -1
    limit: int | None = None
    max_cached_documents: int = 20000
    cache_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "DataConfig":
        """Build a data configuration from environment variables.

        Returns:
            A :class:`DataConfig` populated from ``NANO_RE_*`` variables.
        """
        raw_languages = os.getenv("NANO_RE_LANGUAGES")
        languages = (
            tuple(part.strip() for part in raw_languages.split(",") if part.strip())
            if raw_languages
            else cls.languages
        )
        max_relations = _env_int("NANO_RE_MAX_RELATIONS", 0)
        return cls(
            languages=languages,
            relation_corpus=_env_str("NANO_RE_RELATION_CORPUS", cls.relation_corpus),
            entity_corpus=_env_str("NANO_RE_ENTITY_CORPUS", cls.entity_corpus),
            relation_weight=_env_float(
                "NANO_RE_RELATION_WEIGHT", cls.relation_weight
            ),
            entity_weight=_env_float("NANO_RE_ENTITY_WEIGHT", cls.entity_weight),
            english_relation_corpus=_env_str(
                "NANO_RE_ENGLISH_RELATION_CORPUS", cls.english_relation_corpus
            ),
            english_relation_weight=_env_float(
                "NANO_RE_ENGLISH_RELATION_WEIGHT", cls.english_relation_weight
            ),
            min_relation_count=_env_int(
                "NANO_RE_MIN_RELATION_COUNT", cls.min_relation_count
            ),
            max_relations=max_relations if max_relations > 0 else None,
            train_split=_env_str("NANO_RE_TRAIN_SPLIT", cls.train_split),
            eval_split=_env_str("NANO_RE_EVAL_SPLIT", cls.eval_split),
            max_sequence_length=_env_int(
                "NANO_RE_MAX_SEQUENCE_LENGTH", cls.max_sequence_length
            ),
            max_negative_pairs=_env_int(
                "NANO_RE_MAX_NEGATIVE_PAIRS", cls.max_negative_pairs
            ),
            train_batch_size=_env_int("NANO_RE_TRAIN_BATCH_SIZE", cls.train_batch_size),
            eval_batch_size=_env_int("NANO_RE_EVAL_BATCH_SIZE", cls.eval_batch_size),
            num_workers=_env_int("NANO_RE_NUM_WORKERS", cls.num_workers),
            max_cached_documents=_env_int(
                "NANO_RE_MAX_CACHED_DOCUMENTS", cls.max_cached_documents
            ),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Settings describing the multi-task architecture.

    Attributes:
        backbone_name: Hugging Face identifier of the shared encoder.
        pair_hidden_size: Width of the relation head's hidden layer.
        dropout: Dropout probability applied inside both task heads.
        entity_pooling: Strategy used to pool mention tokens into an entity
            vector. Only ``mean`` is currently implemented.
        localized_context: Whether the relation head receives a context vector
            built from the tokens both entities attend to. It is the second half
            of the adaptive thresholding method and improves relation quality,
            at the cost of materialising the encoder's attention maps, which
            grow with the square of the sequence length.
        trim_vocabulary: Whether to compact the embedding table to the tokens
            the configured languages actually use. It roughly quarters the model,
            but it also destroys performance on every language left out, so it
            defaults to off: a released multilingual model should work in the
            languages it claims. Turn it on for a deployment with a fixed,
            known language set.
        vocabulary_coverage: Fraction of observed token occurrences the trimmed
            vocabulary must still cover.
        min_vocabulary_size: Floor on the trimmed vocabulary, guarding against a
            small sample producing an unusably narrow table.
    """

    backbone_name: str = "jhu-clsp/mmBERT-small"
    pair_hidden_size: int = 512
    dropout: float = 0.1
    entity_pooling: str = "mean"
    localized_context: bool = True
    trim_vocabulary: bool = False
    vocabulary_coverage: float = 0.9999
    min_vocabulary_size: int = 8000

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Build a model configuration from environment variables.

        Returns:
            A :class:`ModelConfig` populated from ``NANO_RE_*`` variables.
        """
        return cls(
            backbone_name=_env_str("NANO_RE_BACKBONE", cls.backbone_name),
            pair_hidden_size=_env_int("NANO_RE_PAIR_HIDDEN_SIZE", cls.pair_hidden_size),
            dropout=_env_float("NANO_RE_DROPOUT", cls.dropout),
            localized_context=_env_bool(
                "NANO_RE_LOCALIZED_CONTEXT", cls.localized_context
            ),
            trim_vocabulary=_env_bool(
                "NANO_RE_TRIM_VOCABULARY", cls.trim_vocabulary
            ),
            vocabulary_coverage=_env_float(
                "NANO_RE_VOCABULARY_COVERAGE", cls.vocabulary_coverage
            ),
            min_vocabulary_size=_env_int(
                "NANO_RE_MIN_VOCABULARY_SIZE", cls.min_vocabulary_size
            ),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Settings controlling the multi-task optimisation loop.

    Attributes:
        epochs: Number of passes over the training split.
        learning_rate: Peak learning rate for the encoder parameters.
        head_learning_rate: Peak learning rate for the randomly initialised heads.
        weight_decay: Decoupled weight decay for non-bias parameters.
        warmup_ratio: Fraction of total steps spent warming the schedule up.
        gradient_accumulation_steps: Micro-batches accumulated per optimiser step.
        max_grad_norm: Threshold for gradient-norm clipping.
        ner_loss_weight: Alpha coefficient of the NER loss term.
        relation_loss_weight: Beta coefficient of the relation loss term.
        relation_loss: Relation loss strategy, ``adaptive_threshold`` or ``bce``.
        relation_threshold: Sigmoid threshold used by the ``bce`` strategy.
        seed: Seed applied to Python, NumPy and PyTorch generators.
        output_dir: Directory receiving checkpoints and reports.
        init_from: Bundle whose weights initialise this run instead of the
            pretrained backbone and fresh heads. This is what chains the two
            stages of a two-phase recipe: pretrain on the automatically
            generated corpus, then fine-tune on a cleaner one from that
            checkpoint.
    """

    epochs: int = 3
    learning_rate: float = 3e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    ner_loss_weight: float = 1.0
    relation_loss_weight: float = 1.0
    relation_loss: str = "adaptive_threshold"
    relation_threshold: float = 0.5
    seed: int = 42
    output_dir: Path = Path("artifacts")
    init_from: Path | None = None

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Build a training configuration from environment variables.

        Returns:
            A :class:`TrainingConfig` populated from ``NANO_RE_*`` variables.
        """
        return cls(
            epochs=_env_int("NANO_RE_EPOCHS", cls.epochs),
            learning_rate=_env_float("NANO_RE_LEARNING_RATE", cls.learning_rate),
            head_learning_rate=_env_float(
                "NANO_RE_HEAD_LEARNING_RATE", cls.head_learning_rate
            ),
            weight_decay=_env_float("NANO_RE_WEIGHT_DECAY", cls.weight_decay),
            warmup_ratio=_env_float("NANO_RE_WARMUP_RATIO", cls.warmup_ratio),
            gradient_accumulation_steps=_env_int(
                "NANO_RE_GRADIENT_ACCUMULATION_STEPS", cls.gradient_accumulation_steps
            ),
            max_grad_norm=_env_float("NANO_RE_MAX_GRAD_NORM", cls.max_grad_norm),
            ner_loss_weight=_env_float("NANO_RE_NER_LOSS_WEIGHT", cls.ner_loss_weight),
            relation_loss_weight=_env_float(
                "NANO_RE_RELATION_LOSS_WEIGHT", cls.relation_loss_weight
            ),
            relation_loss=_env_str("NANO_RE_RELATION_LOSS", cls.relation_loss),
            seed=_env_int("NANO_RE_SEED", cls.seed),
            output_dir=Path(_env_str("NANO_RE_OUTPUT_DIR", str(cls.output_dir))),
            init_from=(
                Path(os.environ["NANO_RE_INIT_FROM"])
                if os.getenv("NANO_RE_INIT_FROM")
                else None
            ),
        )


@dataclass(frozen=True)
class ExportConfig:
    """Settings for ONNX export, quantisation and CPU benchmarking.

    Attributes:
        opset_version: ONNX opset targeted by the exporter.
        fp32_filename: File name of the exported FP32 graph.
        int8_filename: File name of the quantised INT8 graph.
        quantized_op_types: ONNX operator types eligible for weight quantisation.
            ``Gather`` covers the embedding table, which dominates model size.
        parity_tolerance: Maximum absolute logit difference tolerated between the
            PyTorch and ONNX Runtime outputs.
        benchmark_warmup: Untimed iterations executed before measuring.
        benchmark_iterations: Timed iterations used for the latency report.
        benchmark_documents: Number of evaluation documents fed to the benchmark.
        intra_op_num_threads: ONNX Runtime CPU thread count. ``0`` lets the
            runtime choose.
    """

    opset_version: int = 18
    fp32_filename: str = "model.onnx"
    int8_filename: str = "model_int8.onnx"
    quantized_op_types: tuple[str, ...] = ("MatMul", "Gather")
    parity_tolerance: float = 1e-3
    benchmark_warmup: int = 5
    benchmark_iterations: int = 30
    benchmark_documents: int = 32
    intra_op_num_threads: int = 0

    @classmethod
    def from_env(cls) -> "ExportConfig":
        """Build an export configuration from environment variables.

        Returns:
            An :class:`ExportConfig` populated from ``NANO_RE_*`` variables.
        """
        raw_ops = os.getenv("NANO_RE_QUANTIZED_OP_TYPES")
        op_types = (
            tuple(part.strip() for part in raw_ops.split(",") if part.strip())
            if raw_ops
            else cls.quantized_op_types
        )
        return cls(
            opset_version=_env_int("NANO_RE_OPSET_VERSION", cls.opset_version),
            quantized_op_types=op_types,
            parity_tolerance=_env_float(
                "NANO_RE_PARITY_TOLERANCE", cls.parity_tolerance
            ),
            benchmark_warmup=_env_int("NANO_RE_BENCHMARK_WARMUP", cls.benchmark_warmup),
            benchmark_iterations=_env_int(
                "NANO_RE_BENCHMARK_ITERATIONS", cls.benchmark_iterations
            ),
            benchmark_documents=_env_int(
                "NANO_RE_BENCHMARK_DOCUMENTS", cls.benchmark_documents
            ),
            intra_op_num_threads=_env_int(
                "NANO_RE_INTRA_OP_NUM_THREADS", cls.intra_op_num_threads
            ),
        )


@dataclass(frozen=True)
class PackagingConfig:
    """Settings for assembling the finished model bundle on disk.

    Attributes:
        model_name: Name written into the generated model card.
        license_id: SPDX identifier recorded in the model card.
        keep_fp32_graph: Whether to retain the float32 ONNX graph after
            quantisation. It is an intermediate of the quantisation step and
            costs roughly four times the INT8 graph on disk, but it is required
            to re-run the benchmark comparison.
    """

    model_name: str = "nano-relation-extractor"
    license_id: str = "mit"
    keep_fp32_graph: bool = True

    @classmethod
    def from_env(cls) -> "PackagingConfig":
        """Build a packaging configuration from environment variables.

        Returns:
            A :class:`PackagingConfig` populated from ``NANO_RE_*`` variables.
        """
        return cls(
            model_name=_env_str("NANO_RE_MODEL_NAME", cls.model_name),
            license_id=_env_str("NANO_RE_LICENSE", cls.license_id),
            keep_fp32_graph=_env_bool(
                "NANO_RE_KEEP_FP32_GRAPH", cls.keep_fp32_graph
            ),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate configuration passed between pipeline stages.

    Attributes:
        data: Dataset and encoding settings.
        model: Architecture settings.
        training: Optimisation settings.
        export: ONNX and quantisation settings.
        packaging: Local bundle settings.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    packaging: PackagingConfig = field(default_factory=PackagingConfig)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build the aggregate configuration from environment variables.

        Returns:
            A fully populated :class:`PipelineConfig`.
        """
        return cls(
            data=DataConfig.from_env(),
            model=ModelConfig.from_env(),
            training=TrainingConfig.from_env(),
            export=ExportConfig.from_env(),
            packaging=PackagingConfig.from_env(),
        )

    def with_overrides(self, **sections: Any) -> "PipelineConfig":
        """Return a copy of this configuration with replaced sections.

        Args:
            **sections: Mapping of section name to replacement dataclass.

        Returns:
            A new :class:`PipelineConfig` instance.
        """
        return replace(self, **sections)

    @property
    def artifacts_dir(self) -> Path:
        """Directory holding every artifact produced by the pipeline."""
        return self.training.output_dir
