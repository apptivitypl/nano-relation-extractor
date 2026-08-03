"""End-to-end pipeline orchestration.

The pipeline is the composition root: it constructs collaborators, hands them to
each other and sequences the stages. Every stage reads its inputs from the
artifact directory and writes its outputs back there, so stages can be run
individually, resumed, or driven cell by cell from a notebook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .data import DataModule
from .export import (
    BenchmarkReport,
    DynamicInt8Quantizer,
    ExportReport,
    LatencyBenchmark,
    OnnxExporter,
    OnnxInferenceSession,
    OnnxModelAdapter,
    QuantizationReport,
)
from .artifacts import BundleAssembler, BundleReport, ModelCardBuilder
from .models import NanoREModelFactory, count_parameters
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
        """Download the corpus, derive the label schema and report statistics.

        Returns:
            The label schema, also written to the artifact directory.
        """
        module = self.data_module
        schema = module.schema
        self._report(
            f"Label schema: {schema.num_bio_labels} BIO tags, "
            f"{schema.num_relation_labels} relation classes."
        )
        encoded = module.encode_split(self._config.data.eval_split, training=False)
        statistics = module.statistics[self._config.data.eval_split]
        self._report(
            f"Split {statistics.split}: {statistics.encoded_documents} of "
            f"{statistics.raw_documents} documents encoded, "
            f"{statistics.gold_triples} gold triples, recall ceiling "
            f"{statistics.recall_ceiling:.4f}."
        )
        if encoded:
            sample = encoded[0]
            self._report(
                f"First document: {sample.input_ids.shape[0]} tokens, "
                f"{sample.num_entities} entities, {sample.num_pairs} candidate pairs."
            )
        schema.save(self.artifacts_dir / SCHEMA_FILENAME)
        return schema

    def train(self) -> TrainingReport:
        """Train the multi-task model and persist it with its tokenizer.

        Returns:
            The training report, also written to the artifact directory.
        """
        module = self.data_module
        schema = module.schema
        schema.save(self.artifacts_dir / SCHEMA_FILENAME)

        model = self._factory.build(self._config.model, schema)
        self._report(
            f"Model assembled: {count_parameters(model) / 1e6:.1f}M parameters."
        )

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
        self._report(f"Device: {device_manager.describe()}.")

        train_loader = module.train_loader()
        eval_loader = module.eval_loader()
        for statistics in module.statistics.values():
            self._report(
                f"Split {statistics.split}: {statistics.encoded_documents} documents, "
                f"{statistics.gold_triples} gold triples."
            )

        trainer = MultiTaskTrainer(
            model=model,
            criterion=criterion,
            evaluator=MultiTaskEvaluator(schema, criterion, device_manager),
            device_manager=device_manager,
            config=self._config.training,
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
            f"{export_report.opset_version}; maximum deviation "
            f"{max(export_report.max_ner_deviation, export_report.max_relation_deviation):.2e}."
        )

        quantization = DynamicInt8Quantizer(self._config.export).quantize(
            fp32_path, int8_path
        )
        self._report(
            f"Quantised {quantization.source_bytes / 1e6:.1f} MB to "
            f"{quantization.target_bytes / 1e6:.1f} MB "
            f"({quantization.size_reduction:.1%} smaller, preprocessing "
            f"{quantization.preprocessing})."
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
        documents = module.encode_split(self._config.data.eval_split, training=False)
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
            dataset_repo_id=self._config.data.dataset_repo_id,
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
        device_manager = DeviceManager(preference="cpu", use_amp=False)
        adapter = OnnxModelAdapter(
            OnnxInferenceSession(
                model_path,
                intra_op_num_threads=self._config.export.intra_op_num_threads,
            )
        )
        loader = module.build_loader(
            documents, self._config.data.eval_batch_size, shuffle=False
        )
        evaluator = MultiTaskEvaluator(module.schema, criterion, device_manager)
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
