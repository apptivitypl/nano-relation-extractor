"""INT8 dynamic quantisation of the exported graph.

Dynamic quantisation stores weights as INT8 and computes activation scales at
run time, which suits variable length text where calibration data would have to
cover every sequence length.

It is an optimisation, so it carries a correctness obligation: a graph that is
four times smaller and answers differently is not a smaller model, it is a
broken one. Quantisation here therefore tries several configurations and keeps
the first whose predictions still agree with float32, rather than applying one
and reporting the damage afterwards.

The configurations exist because the failure has a documented cause. ONNX
Runtime's dynamic path pairs unsigned activations with signed weights, and on
x86 without VNNI that product is accumulated with ``VPMADDUBSW``, which
saturates at sixteen bits and clamps. The ONNX Runtime documentation names two
remedies for exactly this, unsigned weights or a reduced range, and both are in
the ladder below. Quantising ``Gather`` is attempted first because the
multilingual embedding table is most of the model, but it is also the most
fragile, so it is the first thing given up.
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

DEFAULT_AGREEMENT = 0.98


@dataclass(frozen=True)
class QuantizationRecipe:
    """One way of quantising, and what it trades away.

    Attributes:
        name: Short identifier reported in the quantisation report.
        op_types: Operator types eligible for weight quantisation.
        weight_type: Integer type weights are stored as.
        per_channel: Whether each output channel gets its own scale, which
            helps when weight ranges differ widely between channels.
        reduce_range: Whether to use seven bits, which avoids the accumulator
            saturation that afflicts pre-VNNI x86.
    """

    name: str
    op_types: tuple[str, ...]
    weight_type: QuantType
    per_channel: bool = False
    reduce_range: bool = False


RECIPES: tuple[QuantizationRecipe, ...] = (
    QuantizationRecipe(
        "int8-per-channel", ("MatMul", "Gather"), QuantType.QInt8, per_channel=True
    ),
    QuantizationRecipe(
        "uint8-per-channel", ("MatMul", "Gather"), QuantType.QUInt8, per_channel=True
    ),
    QuantizationRecipe(
        "uint8-matmul", ("MatMul",), QuantType.QUInt8, per_channel=True
    ),
    QuantizationRecipe(
        "int8-matmul-reduced",
        ("MatMul",),
        QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    ),
)
"""Configurations tried in order, from smallest result to most conservative."""


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
        recipe: Name of the configuration that was kept.
        agreement: Fraction of predictions matching float32, or ``None`` when
            no validator was supplied.
        rejected: Configurations tried and discarded, with their agreement.
    """

    source_path: Path
    target_path: Path
    source_bytes: int
    target_bytes: int
    quantized_op_types: tuple[str, ...]
    preprocessing: str
    recipe: str = "int8"
    agreement: float | None = None
    rejected: tuple[tuple[str, float], ...] = ()

    @property
    def is_usable(self) -> bool:
        """Whether the quantised graph agrees with float32 well enough to ship."""
        return self.agreement is None or self.agreement >= DEFAULT_AGREEMENT

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
            "recipe": self.recipe,
            "agreement": self.agreement,
            "is_usable": self.is_usable,
            "rejected": [list(item) for item in self.rejected],
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
            recipe=str(payload.get("recipe", "int8")),
            agreement=(
                float(payload["agreement"])
                if payload.get("agreement") is not None
                else None
            ),
            rejected=tuple(
                (str(name), float(score))
                for name, score in payload.get("rejected", [])
            ),
        )


class DynamicInt8Quantizer:
    """Applies INT8 dynamic weight quantisation to an ONNX graph.

    Args:
        config: Export settings selecting the operator types to quantise.
    """

    def __init__(self, config: ExportConfig) -> None:
        self._config = config

    def quantize(
        self, source: Path, target: Path, validator=None
    ) -> QuantizationReport:
        """Quantise a graph, keeping the smallest result that still agrees.

        Without a validator the first configuration is applied and trusted,
        which is what a caller with no data to check against can do. With one,
        each configuration is measured and the first that clears the agreement
        threshold wins; the rest are recorded so the report shows what was tried.

        Args:
            source: Float32 graph produced by the exporter.
            target: Path of the INT8 graph to write.
            validator: Optional callable taking a graph path and returning the
                fraction of predictions matching float32.

        Returns:
            A report describing both files and which configuration survived.

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

            rejected: list[tuple[str, float]] = []
            best: tuple[QuantizationRecipe, float] | None = None

            for recipe in self._recipes():
                candidate = Path(staging) / f"{recipe.name}.onnx"
                self._apply(model_input, candidate, recipe)
                if validator is None:
                    candidate.replace(target)
                    return self._report(
                        source, target, preprocessing, recipe, None, ()
                    )

                agreement = float(validator(candidate))
                if agreement >= DEFAULT_AGREEMENT:
                    candidate.replace(target)
                    return self._report(
                        source, target, preprocessing, recipe, agreement,
                        tuple(rejected),
                    )
                rejected.append((recipe.name, agreement))
                if best is None or agreement > best[1]:
                    best = (recipe, agreement)
                    candidate.replace(target)

            recipe, agreement = best
            return self._report(
                source, target, preprocessing, recipe, agreement,
                tuple(item for item in rejected if item[0] != recipe.name),
            )

    def _recipes(self) -> tuple[QuantizationRecipe, ...]:
        """Return the configurations to try, honouring an explicit override.

        Returns:
            The ladder, or a single configuration when the operator types were
            set explicitly.
        """
        configured = tuple(self._config.quantized_op_types)
        if configured and configured != ("MatMul", "Gather"):
            return (
                QuantizationRecipe(
                    "configured", configured, QuantType.QInt8, per_channel=True
                ),
            )
        return RECIPES

    def _apply(
        self, source: Path, target: Path, recipe: QuantizationRecipe
    ) -> None:
        """Run one quantisation configuration.

        Args:
            source: Preprocessed float32 graph.
            target: Where to write the quantised graph.
            recipe: Configuration to apply.
        """
        quantize_dynamic(
            model_input=str(source),
            model_output=str(target),
            weight_type=recipe.weight_type,
            op_types_to_quantize=list(recipe.op_types),
            per_channel=recipe.per_channel,
            reduce_range=recipe.reduce_range,
            extra_options={"EnableSubgraph": False},
        )

    def _report(
        self,
        source: Path,
        target: Path,
        preprocessing: str,
        recipe: QuantizationRecipe,
        agreement: float | None,
        rejected: tuple[tuple[str, float], ...],
    ) -> QuantizationReport:
        """Assemble the report for a completed quantisation.

        Args:
            source: Float32 graph.
            target: Quantised graph.
            preprocessing: Stage the graph accepted.
            recipe: Configuration that was kept.
            agreement: Measured agreement, when a validator ran.
            rejected: Configurations discarded along the way.

        Returns:
            The report.
        """
        return QuantizationReport(
            source_path=source,
            target_path=target,
            source_bytes=source.stat().st_size,
            target_bytes=target.stat().st_size,
            quantized_op_types=recipe.op_types,
            preprocessing=preprocessing,
            recipe=recipe.name,
            agreement=agreement,
            rejected=rejected,
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
