"""End-to-end pipeline orchestration.

The pipeline is the composition root: it constructs collaborators, hands them to
each other and sequences the stages. Every stage reads its inputs from the
artifact directory and writes its outputs back there, so stages can be run
individually, resumed, or driven cell by cell from a notebook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .config import PipelineConfig
from .data import DataModule, MultiTaskCollator
from .export import (
    BenchmarkReport,
    DynamicInt8Quantizer,
    ExportReport,
    LatencyBenchmark,
    OnnxExporter,
    OnnxInferenceSession,
    OnnxModelAdapter,
    build_feeds,
    QuantizationReport,
)
from .artifacts import BundleAssembler, BundleReport, ModelCardBuilder
from .models import NanoREModelFactory, count_parameters
from .models.vocabulary import VocabularyTrimmer
from .schema import LabelSchema
from .training import (
    DeviceManager,
    MultiTaskEvaluator,
    MultiTaskLoss,
    MultiTaskTrainer,
    TrainingReport,
    build_relation_objective,
)

SCHEMA_FILENAME = "label_schema.json"
TRAINING_REPORT_FILENAME = "training_report.json"
EXPORT_REPORT_FILENAME = "export_report.json"
BENCHMARK_FILENAME = "benchmark.json"
CARD_FILENAME = "MODEL_CARD.md"


@dataclass
class ExportArtifacts:
    """Reports produced by the export stage.

    Attributes:
        export: Report describing the float32 graph and its verification.
        quantization: Report describing the INT8 graph.
    """

    export: ExportReport
    quantization: QuantizationReport

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of both reports."""
        return {
            "export": self.export.to_dict(),
            "quantization": self.quantization.to_dict(),
        }

    def save(self, path: Path) -> Path:
        """Write both reports to disk as JSON.

        Args:
            path: Destination file path.

        Returns:
            The path that was written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_quantization(cls, path: Path) -> QuantizationReport | None:
        """Read only the quantisation report from a saved file.

        Args:
            path: Source file path.

        Returns:
            The quantisation report, or ``None`` when the file is absent.
        """
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return QuantizationReport.from_dict(payload["quantization"])


class Pipeline:
    """Runs the training, export, benchmark and packaging stages.

    Args:
        config: Aggregate configuration for every stage.
        reporter: Callable receiving progress messages. Defaults to ``print``.
    """

    def __init__(self, config: PipelineConfig, reporter=print) -> None:
        self._config = config
        self._report = reporter
        self._factory = NanoREModelFactory()
        self._data_module: DataModule | None = None

    @property
    def config(self) -> PipelineConfig:
        """Configuration driving every stage."""
        return self._config

    @property
    def artifacts_dir(self) -> Path:
        """Directory holding the artifacts of every stage."""
        return self._config.artifacts_dir

    @property
    def data_module(self) -> DataModule:
        """Data module built from configuration, created on first use."""
        if self._data_module is None:
            self._data_module = DataModule(
                data_config=self._config.data,
                model_config=self._config.model,
            )
        return self._data_module

    def prepare(self) -> LabelSchema:
        """Read the corpora, derive the label schema and report statistics.

        Returns:
            The label schema, also written to the artifact directory.
        """
        module = self.data_module
        self._report(f"Languages: {', '.join(module.languages)}")
        bundle = module.build_corpus(self._config.data.train_split, training=True)
        self._report(bundle.describe())

        schema = module.schema
        counts = module.inventory.counts
        rare = sum(1 for count in counts.values() if count < 10)
        self._report(
            f"Schema: {schema.num_bio_labels} BIO tags, "
            f"{schema.num_relation_labels} relation classes, drawn from "
            f"{len(counts)} observed predicates of which {rare} occur fewer "
            "than ten times."
        )
        for index in range(min(len(bundle.dataset), 50)):
            sample = bundle.dataset[index]
            if sample is not None:
                self._report(
                    f"First document: {sample.input_ids.shape[0]} sub-words, "
                    f"{sample.num_entities} entities, {sample.num_pairs} "
                    "candidate pairs."
                )
                break
        schema.save(self.artifacts_dir / SCHEMA_FILENAME)
        return schema

    def train(self) -> TrainingReport:
        """Train the multi-task model and persist it with its tokenizer.

        Returns:
            The training report, also written to the artifact directory.
        """
        module = self.data_module
        train_bundle = module.build_corpus(
            self._config.data.train_split, training=True
        )
        eval_bundle = module.build_corpus(
            self._config.data.eval_split, training=False
        )
        self._report(train_bundle.describe())
        self._report(eval_bundle.describe())

        schema = module.schema
        schema.save(self.artifacts_dir / SCHEMA_FILENAME)

        model = self._build_model(schema)
        self._report(
            f"Model assembled: {count_parameters(model) / 1e6:.1f}M parameters."
        )
        self._trim_vocabulary(model, module, train_bundle)

        objective = build_relation_objective(
            self._config.training.relation_loss,
            threshold=self._config.training.relation_threshold,
        )
        criterion = MultiTaskLoss(
            relation_objective=objective,
            ner_weight=self._config.training.ner_loss_weight,
            relation_weight=self._config.training.relation_loss_weight,
        )
        device_manager = DeviceManager()
        if device_manager.warning:
            self._report(f"Warning: {device_manager.warning}")
        tuning = device_manager.tuning
        batch_size = self._fit_batch_size(model, criterion, device_manager)
        accumulation = device_manager.resolve_accumulation(
            self._config.training.gradient_accumulation_steps, batch_size
        )
        self._report(
            f"Device: {device_manager.describe()}, batch {batch_size} "
            f"x{accumulation} = {batch_size * accumulation} documents per step."
        )
        training_config = replace(
            self._config.training, gradient_accumulation_steps=accumulation
        )

        train_loader = module.build_loader(
            train_bundle.dataset, batch_size, shuffle=True, tuning=tuning
        )
        eval_loader = module.build_loader(
            eval_bundle.dataset, batch_size, shuffle=False, tuning=tuning
        )

        trainer = MultiTaskTrainer(
            model=device_manager.parallelise(model),
            criterion=criterion,
            evaluator=MultiTaskEvaluator(schema, criterion, device_manager),
            device_manager=device_manager,
            config=training_config,
            on_epoch_end=self._log_epoch,
        )
        report = trainer.train(train_loader, eval_loader)

        self._factory.save(model, self.artifacts_dir)
        module.tokenizer.save_pretrained(str(self.artifacts_dir))
        report.save(self.artifacts_dir / TRAINING_REPORT_FILENAME)
        self._report(
            f"Best epoch {report.best_epoch}, combined F1 "
            f"{report.best_evaluation.combined_f1:.4f}."
            if report.best_evaluation
            else "Training finished without an evaluation result."
        )
        return report

    def export(self) -> ExportArtifacts:
        """Export the trained model to ONNX and quantise it to INT8.

        Returns:
            The export and quantisation reports.
        """
        model = self._factory.load(self.artifacts_dir)
        fp32_path = self.artifacts_dir / self._config.export.fp32_filename
        int8_path = self.artifacts_dir / self._config.export.int8_filename

        export_report = OnnxExporter(self._config.export).export(model, fp32_path)
        self._report(
            f"Exported with the {export_report.exporter} backend at opset "
            f"{export_report.opset_version}; relative deviation "
            f"{export_report.max_relative_deviation:.2e}, decisions match: "
            f"{export_report.decisions_match}."
        )

        quantization = DynamicInt8Quantizer(self._config.export).quantize(
            fp32_path, int8_path, validator=self._build_quantisation_validator(fp32_path)
        )
        self._report(
            f"Quantised {quantization.source_bytes / 1e6:.1f} MB to "
            f"{quantization.target_bytes / 1e6:.1f} MB "
            f"({quantization.size_reduction:.1%} smaller, preprocessing "
            f"{quantization.preprocessing})."
        )
        if quantization.rejected:
            discarded = ", ".join(
                f"{name} ({score:.1%})" for name, score in quantization.rejected
            )
            self._report(f"Quantisation rejected: {discarded}.")
        if not quantization.is_usable:
            self._report(
                "Warning: no INT8 configuration preserved the model's "
                f"predictions (best {quantization.agreement:.1%} agreement). "
                "Deploy model.onnx instead; the INT8 graph is not usable."
            )
        artifacts = ExportArtifacts(export=export_report, quantization=quantization)
        artifacts.save(self.artifacts_dir / EXPORT_REPORT_FILENAME)
        return artifacts

    def benchmark(self, measure_accuracy: bool = True) -> BenchmarkReport:
        """Compare the float32 and INT8 graphs on CPU.

        Args:
            measure_accuracy: Whether to also score both graphs, which reveals
                the accuracy cost of quantisation rather than only its speed
                benefit.

        Returns:
            The benchmark report, also written to the artifact directory.
        """
        module = self.data_module
        module.set_schema(LabelSchema.load(self.artifacts_dir / SCHEMA_FILENAME))
        bundle = module.build_corpus(self._config.data.eval_split, training=False)
        documents = [
            encoded
            for encoded in (
                bundle.dataset[index]
                for index in range(
                    min(len(bundle.dataset), self._config.export.benchmark_documents)
                )
            )
            if encoded is not None
        ]
        fp32_path = self.artifacts_dir / self._config.export.fp32_filename
        int8_path = self.artifacts_dir / self._config.export.int8_filename

        benchmark = LatencyBenchmark(
            self._config.export, module.tokenizer.pad_token_id
        )
        report = benchmark.compare(fp32_path, int8_path, documents)
        self._report(
            f"FP32 {report.fp32.median_ms:.1f} ms/page "
            f"({report.fp32.size_mb:.1f} MB), INT8 {report.int8.median_ms:.1f} "
            f"ms/page ({report.int8.size_mb:.1f} MB), speedup "
            f"{report.speedup:.2f}x, size reduction {report.size_reduction:.1%}."
        )

        if measure_accuracy:
            subset = documents[: self._config.export.benchmark_documents]
            report.fp32_evaluation = self._score_graph(fp32_path, subset)
            report.int8_evaluation = self._score_graph(int8_path, subset)
            self._report(
                f"Accuracy on {len(subset)} documents: NER F1 "
                f"{report.fp32_evaluation.ner.f1:.4f} to "
                f"{report.int8_evaluation.ner.f1:.4f}, relation F1 "
                f"{report.fp32_evaluation.relation.f1:.4f} to "
                f"{report.int8_evaluation.relation.f1:.4f}."
            )

        report.save(self.artifacts_dir / BENCHMARK_FILENAME)
        return report

    def package(
        self,
        training: TrainingReport | None = None,
        benchmark: BenchmarkReport | None = None,
        quantization: QuantizationReport | None = None,
    ) -> BundleReport:
        """Generate the model card and finalise the local bundle.

        Every report falls back to the copy written by its own stage, so
        packaging works both inside a full run and as a standalone command.

        Args:
            training: Training report. Reloaded from disk when omitted.
            benchmark: Benchmark report. Reloaded from disk when omitted.
            quantization: Quantisation report. Reloaded from disk when omitted.

        Returns:
            The bundle inventory.

        Raises:
            FileNotFoundError: If the training report is neither supplied nor
                present in the artifact directory.
        """
        training = training or self._load_training_report()
        if benchmark is None:
            benchmark_path = self.artifacts_dir / BENCHMARK_FILENAME
            benchmark = (
                BenchmarkReport.load(benchmark_path)
                if benchmark_path.exists()
                else None
            )
        if quantization is None:
            quantization = ExportArtifacts.load_quantization(
                self.artifacts_dir / EXPORT_REPORT_FILENAME
            )

        schema = LabelSchema.load(self.artifacts_dir / SCHEMA_FILENAME)
        builder = ModelCardBuilder(
            config=self._config.packaging,
            corpora=tuple(
                name
                for name, weight in (
                    (self._config.data.relation_corpus,
                     self._config.data.relation_weight),
                    (self._config.data.entity_corpus,
                     self._config.data.entity_weight),
                    (self._config.data.english_relation_corpus,
                     self._config.data.english_relation_weight),
                )
                if weight > 0
            ),
        )
        card = builder.build(
            schema=schema,
            training=training,
            benchmark=benchmark,
            quantization=quantization,
            parameter_count=count_parameters(self._factory.load(self.artifacts_dir)),
        )
        builder.write(card, self.artifacts_dir / CARD_FILENAME)

        assembler = BundleAssembler(self._config.packaging)
        report = assembler.assemble(
            self.artifacts_dir,
            fp32_graph=self.artifacts_dir / self._config.export.fp32_filename,
        )
        self._report(report.render())
        if report.is_complete:
            self._report(f"Bundle ready at {self.artifacts_dir.resolve()}.")
        else:
            self._report(
                "Bundle is incomplete. Re-run the stage that produces the "
                "missing artifacts."
            )
        return report

    def run_all(self) -> BundleReport:
        """Run every stage in order.

        Returns:
            The bundle inventory produced by the final stage.
        """
        self.prepare()
        training = self.train()
        artifacts = self.export()
        benchmark = self.benchmark()
        return self.package(
            training=training,
            benchmark=benchmark,
            quantization=artifacts.quantization,
        )

    def _fit_batch_size(self, model, criterion, device_manager) -> int:
        """Find the largest batch that completes one real training step.

        A static table cannot predict this: memory depends on sequence length,
        how many entities a document has and whether context pooling is on. One
        probe of each candidate takes seconds and turns an out of memory failure
        an hour into training into a decision made before it starts.

        Args:
            model: The assembled model, moved to the device by this method.
            criterion: Loss used for the probe, so the backward pass is included.
            device_manager: Resolved device and its tuning.

        Returns:
            The batch size training will use.
        """
        configured = self._config.data.train_batch_size
        model.to(device_manager.device)
        if configured > 0:
            return configured

        sequence = self._config.data.max_sequence_length
        entities, pairs = 24, 128
        vocabulary = int(model.backbone.encoder.config.vocab_size)

        def probe(size: int) -> None:
            """Run one full training step at a candidate batch size."""
            device = device_manager.device
            inputs = torch.randint(0, vocabulary, (size, sequence), device=device)
            mask = torch.ones_like(inputs)
            mention = torch.rand((size, entities, sequence), device=device)
            mention = mention / mention.sum(dim=-1, keepdim=True)
            index = torch.randint(0, entities, (size, pairs, 2), device=device)
            targets = torch.zeros(
                (size, pairs, model.architecture.num_relation_labels), device=device
            )
            targets[..., 0] = 1.0
            with device_manager.autocast():
                outputs = model(
                    input_ids=inputs,
                    attention_mask=mask,
                    mention_mask=mention,
                    pair_index=index,
                )
                losses = criterion(
                    ner_logits=outputs.ner_logits,
                    ner_labels=torch.zeros(
                        (size, sequence), dtype=torch.long, device=device
                    ),
                    relation_logits=outputs.relation_logits,
                    relation_labels=targets,
                    pair_mask=torch.ones((size, pairs), device=device),
                )
            losses.total.backward()
            model.zero_grad(set_to_none=True)

        fitted = device_manager.fit_batch_size(
            probe, device_manager.tuning.batch_size
        )
        device_manager.empty_cache()
        if fitted < device_manager.tuning.batch_size:
            self._report(
                f"Batch reduced from {device_manager.tuning.batch_size} to "
                f"{fitted} to fit available memory."
            )
        return fitted

    def _build_quantisation_validator(self, fp32_path: Path):
        """Build a check that a quantised graph still predicts what float32 does.

        Agreement is measured on real documents rather than random tensors,
        because quantisation error concentrates where activations are largest
        and synthetic input does not put them there.

        Args:
            fp32_path: The float32 graph to compare against.

        Returns:
            A callable taking a graph path and returning the fraction of token
            predictions that match, or ``None`` when no documents are available.
        """
        import numpy as np

        module = self.data_module
        module.set_schema(LabelSchema.load(self.artifacts_dir / SCHEMA_FILENAME))
        bundle = module.build_corpus(self._config.data.eval_split, training=False)
        documents = [
            encoded
            for encoded in (
                bundle.dataset[index]
                for index in range(min(len(bundle.dataset), 16))
            )
            if encoded is not None
        ]
        if not documents:
            return None

        collator = MultiTaskCollator(pad_token_id=module.tokenizer.pad_token_id)
        feeds = [
            build_feeds(
                batch.input_ids,
                batch.attention_mask,
                batch.mention_mask,
                batch.pair_index,
            )
            for batch in (collator([document]) for document in documents)
            if batch is not None
        ]
        reference = OnnxInferenceSession(fp32_path)
        expected = [reference.run(payload) for payload in feeds]

        def validate(candidate: Path) -> float:
            """Return the fraction of predictions matching float32."""
            session = OnnxInferenceSession(candidate)
            matched = total = 0
            for payload, (gold_ner, gold_relation) in zip(feeds, expected):
                ner, relation = session.run(payload)
                matched += int((ner.argmax(-1) == gold_ner.argmax(-1)).sum())
                total += int(np.prod(ner.shape[:-1]))
                gold_pairs = gold_relation > gold_relation[..., :1]
                pairs = relation > relation[..., :1]
                matched += int((pairs == gold_pairs).sum())
                total += int(gold_pairs.size)
            return matched / total if total else 1.0

        return validate

    def _trim_vocabulary(self, model, module, bundle) -> None:
        """Compact the embedding table to the languages actually in scope.

        Trimming runs before training rather than after, so the retained rows
        are the ones that receive gradients. Doing it afterwards would leave the
        kept embeddings tuned for a vocabulary that no longer exists.

        Args:
            model: Model whose backbone is trimmed in place.
            module: Data module supplying the tokenizer.
            bundle: Training corpus whose documents define the token counts.
        """
        if not self._config.model.trim_vocabulary:
            return
        trimmer = VocabularyTrimmer(
            tokenizer=module.tokenizer,
            target_coverage=self._config.model.vocabulary_coverage,
            min_vocab_size=self._config.model.min_vocabulary_size,
        )
        trimmer.observe_documents(bundle.dataset.documents())
        report = trimmer.trim(model.backbone)
        self._report(report.describe())
        self._report(
            f"Model after trimming: {count_parameters(model) / 1e6:.1f}M parameters."
        )

    def _build_model(self, schema: LabelSchema):
        """Assemble the model, optionally resuming from an earlier stage.

        Args:
            schema: Label vocabularies determining both head widths.

        Returns:
            A freshly assembled model, or one restored from ``init_from``.

        Raises:
            FileNotFoundError: If ``init_from`` names a directory with no
                checkpoint.
            ValueError: If the checkpoint's relation vocabulary does not match
                the current schema, which would silently mis-map every label.
        """
        init_from = self._config.training.init_from
        if init_from is None:
            return self._factory.build(self._config.model, schema)

        model = self._factory.load(Path(init_from))
        architecture = model.architecture
        if architecture.num_relation_labels != schema.num_relation_labels or (
            architecture.num_bio_labels != schema.num_bio_labels
        ):
            raise ValueError(
                f"Checkpoint at {init_from} has "
                f"{architecture.num_bio_labels} BIO and "
                f"{architecture.num_relation_labels} relation labels, but the "
                f"current schema has {schema.num_bio_labels} and "
                f"{schema.num_relation_labels}. Both stages must use the same "
                "label schema."
            )
        self._report(f"Initialised from the checkpoint at {init_from}.")
        return model

    def _load_training_report(self) -> TrainingReport:
        """Read the training report written by the training stage.

        Returns:
            The reconstructed report.

        Raises:
            FileNotFoundError: If the report is absent.
        """
        path = self.artifacts_dir / TRAINING_REPORT_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run the training stage before packaging."
            )
        return TrainingReport.load(path)

    def _score_graph(self, model_path: Path, documents):
        """Evaluate one ONNX graph on a subset of documents.

        Args:
            model_path: Graph to score.
            documents: Encoded documents to score against.

        Returns:
            The evaluation result for that graph.
        """
        module = self.data_module
        objective = build_relation_objective(
            self._config.training.relation_loss,
            threshold=self._config.training.relation_threshold,
        )
        criterion = MultiTaskLoss(
            relation_objective=objective,
            ner_weight=self._config.training.ner_loss_weight,
            relation_weight=self._config.training.relation_loss_weight,
        )
        schema = module.schema
        device_manager = DeviceManager(preference="cpu", use_amp=False)
        adapter = OnnxModelAdapter(
            OnnxInferenceSession(
                model_path,
                intra_op_num_threads=self._config.export.intra_op_num_threads,
            )
        )
        loader = module.build_loader(
            documents,
            device_manager.resolve_batch_size(self._config.data.eval_batch_size),
            shuffle=False,
        )
        evaluator = MultiTaskEvaluator(schema, criterion, device_manager)
        return evaluator.evaluate(adapter, loader)

    def _log_epoch(self, epoch) -> None:
        """Report the outcome of one training epoch.

        Args:
            epoch: The epoch report emitted by the trainer.
        """
        evaluation = epoch.evaluation
        self._report(
            f"Epoch {epoch.epoch}: loss {epoch.train_loss:.4f} "
            f"(NER {epoch.train_ner_loss:.4f}, RE {epoch.train_relation_loss:.4f}) "
            f"| NER F1 {evaluation.ner.f1:.4f} | RE F1 {evaluation.relation.f1:.4f} "
            f"| {epoch.seconds:.1f}s"
        )
