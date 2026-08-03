"""CPU latency and accuracy comparison between the FP32 and INT8 graphs.

Latency is measured one document per call, because the deployment unit is a
page rather than a batch, and reported as median and ninety-fifth percentile.
A mean alone hides the tail that a service level objective is written against.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ExportConfig
from ..data import EncodedDocument, MultiTaskCollator
from ..training.metrics import EvaluationResult
from .runtime import OnnxInferenceSession, build_feeds


@dataclass(frozen=True)
class LatencyMeasurement:
    """Latency and size statistics for one graph.

    Attributes:
        label: Human readable name of the measured graph.
        path: Location of the graph.
        size_bytes: Size of the graph on disk.
        median_ms: Median milliseconds per document.
        mean_ms: Mean milliseconds per document.
        p95_ms: Ninety-fifth percentile milliseconds per document.
        iterations: Number of timed measurements.
    """

    label: str
    path: Path
    size_bytes: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    iterations: int

    @property
    def size_mb(self) -> float:
        """Size of the graph in megabytes."""
        return self.size_bytes / 1e6

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the measurement."""
        return {
            "label": self.label,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_mb": self.size_mb,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "iterations": self.iterations,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LatencyMeasurement":
        """Rebuild a measurement from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed measurement.
        """
        return cls(
            label=str(payload["label"]),
            path=Path(str(payload["path"])),
            size_bytes=int(payload["size_bytes"]),
            median_ms=float(payload["median_ms"]),
            mean_ms=float(payload["mean_ms"]),
            p95_ms=float(payload["p95_ms"]),
            iterations=int(payload["iterations"]),
        )


@dataclass
class BenchmarkReport:
    """Comparison between the float32 and INT8 graphs.

    Attributes:
        fp32: Latency and size of the float32 graph.
        int8: Latency and size of the INT8 graph.
        fp32_evaluation: Accuracy of the float32 graph, when measured.
        int8_evaluation: Accuracy of the INT8 graph, when measured.
        threads: ONNX Runtime thread setting used, ``0`` meaning automatic.
        documents: Number of distinct documents fed to the benchmark.
    """

    fp32: LatencyMeasurement
    int8: LatencyMeasurement
    fp32_evaluation: EvaluationResult | None = None
    int8_evaluation: EvaluationResult | None = None
    threads: int = 0
    documents: int = 0

    @property
    def speedup(self) -> float:
        """Median latency of the float32 graph divided by the INT8 graph."""
        if not self.int8.median_ms:
            return 0.0
        return self.fp32.median_ms / self.int8.median_ms

    @property
    def size_reduction(self) -> float:
        """Fraction of the float32 file size eliminated by quantisation."""
        if not self.fp32.size_bytes:
            return 0.0
        return 1.0 - (self.int8.size_bytes / self.fp32.size_bytes)

    @property
    def relation_f1_delta(self) -> float | None:
        """Relation F1 lost to quantisation, when both graphs were scored."""
        if self.fp32_evaluation is None or self.int8_evaluation is None:
            return None
        return self.int8_evaluation.relation.f1 - self.fp32_evaluation.relation.f1

    @property
    def ner_f1_delta(self) -> float | None:
        """NER F1 lost to quantisation, when both graphs were scored."""
        if self.fp32_evaluation is None or self.int8_evaluation is None:
            return None
        return self.int8_evaluation.ner.f1 - self.fp32_evaluation.ner.f1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the report."""
        return {
            "fp32": self.fp32.to_dict(),
            "int8": self.int8.to_dict(),
            "fp32_evaluation": (
                self.fp32_evaluation.to_dict() if self.fp32_evaluation else None
            ),
            "int8_evaluation": (
                self.int8_evaluation.to_dict() if self.int8_evaluation else None
            ),
            "speedup": self.speedup,
            "size_reduction": self.size_reduction,
            "ner_f1_delta": self.ner_f1_delta,
            "relation_f1_delta": self.relation_f1_delta,
            "threads": self.threads,
            "documents": self.documents,
        }

    def save(self, path: Path) -> Path:
        """Write the report to disk as JSON.

        Args:
            path: Destination file path.

        Returns:
            The path that was written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BenchmarkReport":
        """Rebuild a benchmark report from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed report.
        """
        fp32_evaluation = payload.get("fp32_evaluation")
        int8_evaluation = payload.get("int8_evaluation")
        return cls(
            fp32=LatencyMeasurement.from_dict(payload["fp32"]),
            int8=LatencyMeasurement.from_dict(payload["int8"]),
            fp32_evaluation=(
                None
                if fp32_evaluation is None
                else EvaluationResult.from_dict(fp32_evaluation)
            ),
            int8_evaluation=(
                None
                if int8_evaluation is None
                else EvaluationResult.from_dict(int8_evaluation)
            ),
            threads=int(payload["threads"]),
            documents=int(payload["documents"]),
        )

    @classmethod
    def load(cls, path: Path) -> "BenchmarkReport":
        """Read a previously written report.

        Args:
            path: Source file path.

        Returns:
            The reconstructed report.
        """
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class LatencyBenchmark:
    """Measures single document CPU latency for an ONNX graph.

    Args:
        config: Export settings controlling warmup, iterations and threads.
        pad_token_id: Padding identifier used to build single document batches.
    """

    def __init__(self, config: ExportConfig, pad_token_id: int) -> None:
        self._config = config
        self._collator = MultiTaskCollator(pad_token_id=pad_token_id)

    def measure(
        self, label: str, model_path: Path, documents: list[EncodedDocument]
    ) -> LatencyMeasurement:
        """Time a graph over a fixed set of documents.

        Args:
            label: Name recorded in the measurement.
            model_path: Graph to time.
            documents: Documents cycled through during measurement.

        Returns:
            The latency and size measurement.

        Raises:
            ValueError: If no documents were supplied.
        """
        if not documents:
            raise ValueError("At least one document is required to benchmark.")

        session = OnnxInferenceSession(
            model_path, intra_op_num_threads=self._config.intra_op_num_threads
        )
        sample = documents[: self._config.benchmark_documents]
        feeds = [self._build_feeds(document) for document in sample]

        for index in range(self._config.benchmark_warmup):
            session.run(feeds[index % len(feeds)])

        durations: list[float] = []
        for index in range(self._config.benchmark_iterations):
            payload = feeds[index % len(feeds)]
            start = time.perf_counter()
            session.run(payload)
            durations.append((time.perf_counter() - start) * 1000.0)

        values = np.asarray(durations, dtype=np.float64)
        return LatencyMeasurement(
            label=label,
            path=model_path,
            size_bytes=session.size_bytes,
            median_ms=float(np.median(values)),
            mean_ms=float(values.mean()),
            p95_ms=float(np.percentile(values, 95)),
            iterations=len(durations),
        )

    def compare(
        self,
        fp32_path: Path,
        int8_path: Path,
        documents: list[EncodedDocument],
    ) -> BenchmarkReport:
        """Time both graphs over the same documents.

        Args:
            fp32_path: Float32 graph.
            int8_path: INT8 graph.
            documents: Documents cycled through during measurement.

        Returns:
            A report comparing latency and file size.
        """
        return BenchmarkReport(
            fp32=self.measure("fp32", fp32_path, documents),
            int8=self.measure("int8", int8_path, documents),
            threads=self._config.intra_op_num_threads,
            documents=min(len(documents), self._config.benchmark_documents),
        )

    def _build_feeds(self, document: EncodedDocument) -> dict[str, np.ndarray]:
        """Turn a single document into ONNX Runtime feeds.

        Args:
            document: Encoded document to convert.

        Returns:
            Mapping of graph input name to array.
        """
        batch = self._collator([document])
        return build_feeds(
            batch.input_ids,
            batch.attention_mask,
            batch.mention_mask,
            batch.pair_index,
        )
