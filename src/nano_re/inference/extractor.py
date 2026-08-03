"""End-to-end extraction from raw text.

The relation head is trained on gold entity clusters supplied through
``mention_mask``, which raw text does not provide. Extraction therefore runs the
model twice: once to tag entities, and once more with the pooling weights built
from those predictions. The second pass is unavoidable, because the input to the
relation head depends on the output of the NER head.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..config import PipelineConfig
from ..schema import LabelSchema
from ..training.losses import RelationObjective, build_relation_objective
from ..patterns import PatternExtractor
from .backends import InferenceBackend, OnnxBackend, TorchBackend
from .chunking import ResultMerger, TextChunker, Window
from .clusterer import MentionClusterer, SurfaceFormClusterer
from .decoder import BioSpanDecoder
from .results import (
    ExtractionResult,
    PredictedEntity,
    PredictedMention,
    PredictedRelation,
)
from .text import WordTokenizer

SCHEMA_FILENAME = "label_schema.json"


@dataclass(frozen=True)
class ExtractionSettings:
    """Knobs controlling a single extraction.

    Attributes:
        max_sequence_length: Maximum number of sub-word tokens considered.
        max_entities: Cap on entity clusters scored for relations. The candidate
            pair count grows quadratically, so a runaway NER output on noisy
            input would otherwise dominate the runtime.
        min_confidence: Lowest confidence reported. Predictions below the
            objective's own decision boundary are never reported regardless.
        top_k: Cap on reported relations, ``0`` meaning no cap.
        overlap: Fraction of each window repeated in the next, so an entity on a
            boundary is seen whole by at least one window.
        extract_patterns: Whether to run the deterministic identifier rules
            alongside the model.
    """

    max_sequence_length: int = 512
    max_entities: int = 64
    min_confidence: float = 0.5
    top_k: int = 0
    overlap: float = 0.25
    extract_patterns: bool = True


class RelationExtractor:
    """Extracts entities and relations from raw text.

    Args:
        tokenizer: Tokenizer matching the trained encoder.
        schema: Label vocabularies of the trained model.
        backend: Execution backend running the model.
        objective: Relation decision rule, which must match the one the model
            was trained with.
        clusterer: Strategy grouping mentions into entities.
        settings: Per-extraction limits.
        patterns: Deterministic identifier extractor. Its matches are reported
            as entities but never scored for relations, because the relation
            head has no training signal for identifier types.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        schema: LabelSchema,
        backend: InferenceBackend,
        objective: RelationObjective,
        clusterer: MentionClusterer | None = None,
        settings: ExtractionSettings | None = None,
        patterns: PatternExtractor | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._schema = schema
        self._backend = backend
        self._objective = objective
        self._clusterer = clusterer or SurfaceFormClusterer()
        self._settings = settings or ExtractionSettings()
        self._words = WordTokenizer()
        self._decoder = BioSpanDecoder(schema)
        self._patterns = patterns or PatternExtractor()
        self._chunker = TextChunker(
            tokenizer=tokenizer,
            max_sequence_length=self._settings.max_sequence_length,
            overlap=self._settings.overlap,
        )
        self._merger = ResultMerger()

    @classmethod
    def from_bundle(
        cls,
        directory: Path,
        backend: str = "onnx-int8",
        config: PipelineConfig | None = None,
        settings: ExtractionSettings | None = None,
    ) -> "RelationExtractor":
        """Build an extractor from a packaged bundle directory.

        Args:
            directory: Bundle produced by the packaging stage.
            backend: One of ``onnx-int8``, ``onnx-fp32`` or ``pytorch``.
            config: Configuration supplying the relation objective and thread
                count. Defaults to the environment configuration.
            settings: Per-extraction limits.

        Returns:
            The configured extractor.

        Raises:
            FileNotFoundError: If the bundle is missing required artifacts.
            ValueError: If the backend name is unknown.
        """
        directory = Path(directory)
        schema_path = directory / SCHEMA_FILENAME
        if not schema_path.exists():
            raise FileNotFoundError(
                f"{schema_path} does not exist. Train and package a model "
                "before extracting."
            )
        config = config or PipelineConfig.from_env()
        return cls(
            tokenizer=AutoTokenizer.from_pretrained(str(directory)),
            schema=LabelSchema.load(schema_path),
            backend=cls._build_backend(directory, backend, config),
            objective=build_relation_objective(
                config.training.relation_loss,
                threshold=config.training.relation_threshold,
            ),
            settings=settings
            or ExtractionSettings(
                max_sequence_length=config.data.max_sequence_length
            ),
        )

    @property
    def backend_name(self) -> str:
        """Identifier of the active execution backend."""
        return self._backend.name

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities, identifiers and relations from a text of any length.

        The text is split into overlapping windows sized against the encoder's
        sub-word budget, each window is extracted independently, and the results
        are merged so an entity crossing a boundary stays one entity.

        Args:
            text: Raw input text.

        Returns:
            The extraction result, with document-global mention offsets.
        """
        words = self._words.split(text)
        if not words:
            return ExtractionResult(words=(), entities=(), relations=())

        windows = self._chunker.split(words)
        per_window = [
            (window, self.extract_window(words[window.start : window.end]))
            for window in windows
        ]
        merged = self._merger.merge(per_window, tuple(words))

        if not self._settings.extract_patterns:
            return merged
        return self._with_patterns(merged, text)

    def extract_window(self, words: list[str]) -> ExtractionResult:
        """Extract from a single window that already fits the encoder.

        Args:
            words: Tokenised window.

        Returns:
            The extraction result, with window-local mention offsets.
        """
        if not words:
            return ExtractionResult(words=(), entities=(), relations=())

        encoding = self._tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self._settings.max_sequence_length,
            return_tensors=None,
        )
        word_ids = encoding.word_ids(0)
        word_to_subwords = self._align(word_ids)
        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([encoding["attention_mask"]], dtype=torch.long)
        sequence_length = input_ids.shape[1]
        encoded_words = max(word_to_subwords) + 1 if word_to_subwords else 0
        truncated = len(words) - encoded_words

        ner_logits, _ = self._backend.run(
            input_ids,
            attention_mask,
            torch.zeros((1, 1, sequence_length), dtype=torch.float32),
            torch.zeros((1, 1, 2), dtype=torch.long),
        )
        mentions = self._decoder.decode(ner_logits[0], word_to_subwords, words)
        entities = tuple(self._clusterer.cluster(mentions))[
            : self._settings.max_entities
        ]

        relations: tuple[PredictedRelation, ...] = ()
        if len(entities) >= 2:
            relations = self._extract_relations(
                entities, word_to_subwords, input_ids, attention_mask, sequence_length
            )

        return ExtractionResult(
            words=tuple(words),
            entities=entities,
            relations=relations,
            truncated_words=truncated,
        )

    def _with_patterns(
        self, result: ExtractionResult, text: str
    ) -> ExtractionResult:
        """Add rule-matched identifiers to a model result.

        Identifiers are appended as entities carrying no relations. Where a rule
        match covers words the model also tagged, the rule wins: a checksummed
        tax identifier is a stronger claim than a tagger's guess.

        Args:
            result: Result produced by the model.
            text: The original text the result came from.

        Returns:
            The combined result.
        """
        matches = self._patterns.align(self._patterns.extract(text), text)
        if not matches:
            return result

        claimed = {
            position
            for match in matches
            for position in range(match.word_start, match.word_end)
        }
        entities: list[PredictedEntity] = []
        remap: dict[int, int] = {}
        for entity in result.entities:
            kept = tuple(
                mention
                for mention in entity.mentions
                if not any(
                    position in claimed
                    for position in range(mention.start, mention.end)
                )
            )
            if not kept:
                continue
            remap[entity.index] = len(entities)
            entities.append(
                PredictedEntity(
                    index=len(entities),
                    name=entity.name,
                    entity_type=entity.entity_type,
                    mentions=kept,
                )
            )

        for match in matches:
            entities.append(
                PredictedEntity(
                    index=len(entities),
                    name=match.value,
                    entity_type=match.entity_type,
                    mentions=(
                        PredictedMention(
                            text=match.text,
                            start=match.word_start,
                            end=match.word_end,
                        ),
                    ),
                )
            )

        relations = tuple(
            PredictedRelation(
                head=remap[relation.head],
                tail=remap[relation.tail],
                relation=relation.relation,
                label=relation.label,
                confidence=relation.confidence,
            )
            for relation in result.relations
            if relation.head in remap and relation.tail in remap
        )
        return ExtractionResult(
            words=result.words,
            entities=tuple(entities),
            relations=relations,
            truncated_words=result.truncated_words,
        )

    def _extract_relations(
        self,
        entities: tuple[PredictedEntity, ...],
        word_to_subwords: dict[int, list[int]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_length: int,
    ) -> tuple[PredictedRelation, ...]:
        """Score every ordered pair of predicted entities.

        Args:
            entities: Predicted entity clusters.
            word_to_subwords: Mapping from word index to sub-word positions.
            input_ids: Sub-word identifiers, shape ``[1, S]``.
            attention_mask: Padding mask, shape ``[1, S]``.
            sequence_length: Number of sub-word positions.

        Returns:
            Predicted relations ordered by descending confidence.
        """
        mention_mask = self._build_mention_mask(
            entities, word_to_subwords, sequence_length
        )
        pairs = [
            (head, tail)
            for head in range(len(entities))
            for tail in range(len(entities))
            if head != tail
        ]
        pair_index = torch.tensor([pairs], dtype=torch.long)

        _, relation_logits = self._backend.run(
            input_ids, attention_mask, mention_mask, pair_index
        )
        predictions = self._objective.decode(relation_logits)[0]
        confidences = self._objective.confidence(relation_logits)[0]

        id_to_relation = self._schema.id_to_relation
        results: list[PredictedRelation] = []
        for row, (head, tail) in enumerate(pairs):
            for column in torch.nonzero(predictions[row]).flatten().tolist():
                confidence = float(confidences[row, column])
                if confidence < self._settings.min_confidence:
                    continue
                relation_id = id_to_relation[int(column)]
                results.append(
                    PredictedRelation(
                        head=head,
                        tail=tail,
                        relation=relation_id,
                        label=self._schema.describe_relation(relation_id),
                        confidence=confidence,
                    )
                )

        results.sort(key=lambda item: item.confidence, reverse=True)
        if self._settings.top_k:
            results = results[: self._settings.top_k]
        return tuple(results)

    def _build_mention_mask(
        self,
        entities: tuple[PredictedEntity, ...],
        word_to_subwords: dict[int, list[int]],
        sequence_length: int,
    ) -> torch.Tensor:
        """Build row-normalised pooling weights for the predicted entities.

        Args:
            entities: Predicted entity clusters.
            word_to_subwords: Mapping from word index to sub-word positions.
            sequence_length: Number of sub-word positions.

        Returns:
            A ``[1, E, S]`` float tensor whose rows sum to one.
        """
        mask = torch.zeros((1, len(entities), sequence_length), dtype=torch.float32)
        for row, entity in enumerate(entities):
            for mention in entity.mentions:
                for word in range(mention.start, mention.end):
                    for position in word_to_subwords.get(word, ()):
                        mask[0, row, position] = 1.0
        totals = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return mask / totals

    def _align(self, word_ids: list[int | None]) -> dict[int, list[int]]:
        """Group sub-word positions by their source word index.

        Args:
            word_ids: Per-position word index, ``None`` for special tokens.

        Returns:
            Mapping from word index to the sub-word positions covering it.
        """
        alignment: dict[int, list[int]] = {}
        for position, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            alignment.setdefault(word_id, []).append(position)
        return alignment

    @staticmethod
    def _build_backend(
        directory: Path, backend: str, config: PipelineConfig
    ) -> InferenceBackend:
        """Instantiate the requested execution backend.

        Args:
            directory: Bundle produced by the packaging stage.
            backend: Backend name.
            config: Configuration supplying the ONNX thread count.

        Returns:
            The configured backend.

        Raises:
            ValueError: If the backend name is unknown.
        """
        if backend == "pytorch":
            return TorchBackend.from_bundle(directory)
        if backend == "onnx-int8":
            return OnnxBackend.from_bundle(
                directory,
                filename=config.export.int8_filename,
                intra_op_num_threads=config.export.intra_op_num_threads,
            )
        if backend == "onnx-fp32":
            return OnnxBackend.from_bundle(
                directory,
                filename=config.export.fp32_filename,
                intra_op_num_threads=config.export.intra_op_num_threads,
            )
        raise ValueError(
            f"Unknown backend {backend!r}. Expected 'onnx-int8', 'onnx-fp32' "
            "or 'pytorch'."
        )
