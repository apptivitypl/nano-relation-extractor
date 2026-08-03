"""INT8 dynamic quantisation of the exported graph.

Dynamic quantisation stores weights as INT8 and computes activation scales at
run time, which suits variable length text where calibration data would have to
cover every sequence length.

Quantising ``Gather`` matters more than usual for this model: the multilingual
vocabulary contributes roughly ninety percent of the parameters, and leaving the
embedding table in float32 caps the achievable size reduction at a few percent.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from ..config import ExportConfig

FULL = "full"
ONNX_SHAPE_ONLY = "onnx_shape_only"
SKIPPED = "skipped"


@dataclass(frozen=True)
class QuantizationReport:
    """Outcome of a quantisation pass.

    Attributes:
        source_path: Graph that was quantised.
        target_path: Quantised graph.
        source_bytes: Size of the float32 graph on disk.
        target_bytes: Size of the INT8 graph on disk.
        quantized_op_types: Operator types that were eligible for quantisation.
        preprocessing: Which preprocessing stage the graph accepted. One of
            ``full``, ``onnx_shape_only`` or ``skipped``.
    """

    source_path: Path
    target_path: Path
    source_bytes: int
    target_bytes: int
    quantized_op_types: tuple[str, ...]
    preprocessing: str

    @property
    def size_reduction(self) -> float:
        """Fraction of the original file size that was eliminated."""
        if not self.source_bytes:
            return 0.0
        return 1.0 - (self.target_bytes / self.source_bytes)

    @property
    def compression_ratio(self) -> float:
        """Ratio of float32 size to INT8 size."""
        if not self.target_bytes:
            return 0.0
        return self.source_bytes / self.target_bytes

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the report."""
        return {
            "source_path": str(self.source_path),
            "target_path": str(self.target_path),
            "source_bytes": self.source_bytes,
            "target_bytes": self.target_bytes,
            "size_reduction": self.size_reduction,
            "compression_ratio": self.compression_ratio,
            "quantized_op_types": list(self.quantized_op_types),
            "preprocessing": self.preprocessing,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "QuantizationReport":
        """Rebuild a quantisation report from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed report.
        """
        return cls(
            source_path=Path(str(payload["source_path"])),
            target_path=Path(str(payload["target_path"])),
            source_bytes=int(payload["source_bytes"]),
            target_bytes=int(payload["target_bytes"]),
            quantized_op_types=tuple(payload["quantized_op_types"]),
            preprocessing=str(payload["preprocessing"]),
        )


class DynamicInt8Quantizer:
    """Applies INT8 dynamic weight quantisation to an ONNX graph.

    Args:
        config: Export settings selecting the operator types to quantise.
    """

    def __init__(self, config: ExportConfig) -> None:
        self._config = config

    def quantize(self, source: Path, target: Path) -> QuantizationReport:
        """Quantise a graph and report the size reduction.

        Args:
            source: Float32 graph produced by the exporter.
            target: Path of the INT8 graph to write.

        Returns:
            A report describing both files.

        Raises:
            FileNotFoundError: If the source graph does not exist.
        """
        if not source.exists():
            raise FileNotFoundError(
                f"{source} does not exist. Run the export stage first."
            )
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as staging:
            prepared = Path(staging) / "prepared.onnx"
            preprocessing = self._preprocess(source, prepared)
            model_input = prepared if preprocessing != SKIPPED else source
            quantize_dynamic(
                model_input=str(model_input),
                model_output=str(target),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=list(self._config.quantized_op_types),
                extra_options={"EnableSubgraph": False},
            )

        return QuantizationReport(
            source_path=source,
            target_path=target,
            source_bytes=source.stat().st_size,
            target_bytes=target.stat().st_size,
            quantized_op_types=self._config.quantized_op_types,
            preprocessing=preprocessing,
        )

    def _preprocess(self, source: Path, prepared: Path) -> str:
        """Run shape inference and graph optimisation ahead of quantisation.

        Preprocessing lets the quantiser resolve tensor shapes it would
        otherwise skip. Symbolic shape inference is attempted first and is
        allowed to fail: transformer encoders build position identifiers with a
        ``Range`` node whose limit is symbolic under a dynamic sequence axis,
        which the inference pass cannot evaluate. Dropping only that stage keeps
        ONNX shape inference and graph optimisation, both of which still help.

        Args:
            source: Graph to preprocess.
            prepared: Destination of the preprocessed graph.

        Returns:
            The name of the stage that succeeded.
        """
        attempts = (
            (FULL, {"skip_symbolic_shape": False}),
            (ONNX_SHAPE_ONLY, {"skip_symbolic_shape": True}),
        )
        for name, options in attempts:
            try:
                quant_pre_process(
                    input_model=str(source),
                    output_model_path=str(prepared),
                    skip_optimization=False,
                    skip_onnx_shape=False,
                    **options,
                )
            except Exception:
                continue
            if prepared.exists():
                return name
        return SKIPPED
