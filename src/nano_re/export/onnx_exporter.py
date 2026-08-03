"""ONNX export with verified dynamic axes.

An export that silently freezes a shape produces a graph that works on the
sample batch and fails on everything else. The exporter therefore treats
verification as part of the export: it compares logits against PyTorch and then
repeats the comparison with different batch, sequence, entity and pair counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import ExportConfig
from ..models import NanoREModel, OnnxExportWrapper

INPUT_NAMES = ("input_ids", "attention_mask", "mention_mask", "pair_index")
OUTPUT_NAMES = ("ner_logits", "relation_logits")

DYNAMIC_AXES: dict[str, dict[int, str]] = {
    "input_ids": {0: "batch", 1: "sequence"},
    "attention_mask": {0: "batch", 1: "sequence"},
    "mention_mask": {0: "batch", 1: "entities", 2: "sequence"},
    "pair_index": {0: "batch", 1: "pairs"},
    "ner_logits": {0: "batch", 1: "sequence"},
    "relation_logits": {0: "batch", 1: "pairs"},
}


class ExportVerificationError(RuntimeError):
    """Raised when the exported graph disagrees with the PyTorch model."""


@dataclass(frozen=True)
class ExportReport:
    """Outcome of an ONNX export.

    Attributes:
        path: Location of the exported graph.
        exporter: Which export backend produced the graph.
        opset_version: ONNX opset the graph targets.
        max_ner_deviation: Largest absolute NER logit difference observed.
        max_relation_deviation: Largest absolute relation logit difference.
        max_relative_deviation: Largest deviation as a fraction of the logit
            range, which is the figure the tolerance is applied to.
        decisions_match: Whether both graphs pick the same argmax everywhere.
        dynamic_shapes_verified: Whether a second batch with different batch,
            sequence, entity and pair counts also matched.
        size_bytes: Size of the exported file on disk.
    """

    path: Path
    exporter: str
    opset_version: int
    max_ner_deviation: float
    max_relation_deviation: float
    max_relative_deviation: float
    decisions_match: bool
    dynamic_shapes_verified: bool
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the report."""
        return {
            "path": str(self.path),
            "exporter": self.exporter,
            "opset_version": self.opset_version,
            "max_ner_deviation": self.max_ner_deviation,
            "max_relation_deviation": self.max_relation_deviation,
            "max_relative_deviation": self.max_relative_deviation,
            "decisions_match": self.decisions_match,
            "dynamic_shapes_verified": self.dynamic_shapes_verified,
            "size_bytes": self.size_bytes,
        }


class SampleInputFactory:
    """Builds synthetic model inputs with arbitrary shapes.

    Synthetic inputs keep the exporter independent of the dataset, so export and
    verification can run against any checkpoint without downloading a corpus.

    Args:
        vocab_size: Upper bound for sampled token identifiers.
        seed: Seed for the random generator.
    """

    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        self._vocab_size = vocab_size
        self._generator = torch.Generator().manual_seed(seed)

    def build(
        self, batch: int, sequence: int, entities: int, pairs: int
    ) -> dict[str, torch.Tensor]:
        """Create one set of model inputs.

        Args:
            batch: Number of documents.
            sequence: Number of sub-word positions.
            entities: Number of entity slots.
            pairs: Number of candidate pairs.

        Returns:
            Keyword arguments accepted by the model's forward method.
        """
        input_ids = torch.randint(
            low=0,
            high=self._vocab_size,
            size=(batch, sequence),
            generator=self._generator,
            dtype=torch.long,
        )
        mention_mask = torch.rand(
            (batch, entities, sequence), generator=self._generator
        )
        mention_mask = mention_mask / mention_mask.sum(dim=-1, keepdim=True)
        pair_index = torch.randint(
            low=0,
            high=max(1, entities),
            size=(batch, pairs, 2),
            generator=self._generator,
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((batch, sequence), dtype=torch.long),
            "mention_mask": mention_mask,
            "pair_index": pair_index,
        }


class OnnxExporter:
    """Exports a trained model to ONNX and verifies the result.

    Args:
        config: Export settings including opset and parity tolerance.
    """

    def __init__(self, config: ExportConfig) -> None:
        self._config = config

    def export(self, model: NanoREModel, destination: Path) -> ExportReport:
        """Export a model and verify it against PyTorch.

        Args:
            model: Trained model to export.
            destination: Path of the ``.onnx`` file to write.

        Returns:
            A report describing the export and its verification.

        Raises:
            ExportVerificationError: If the exported graph deviates beyond the
                configured tolerance or rejects a differently shaped batch.
        """
        model = model.eval().to(torch.device("cpu"))
        wrapper = OnnxExportWrapper(model).eval()
        factory = SampleInputFactory(
            vocab_size=int(model.backbone.encoder.config.vocab_size)
        )
        sample = factory.build(batch=2, sequence=32, entities=4, pairs=6)

        destination.parent.mkdir(parents=True, exist_ok=True)
        exporter = self._write_graph(wrapper, sample, destination)

        checks = [
            self._compare(wrapper, destination, sample),
            self._compare(
                wrapper,
                destination,
                factory.build(batch=1, sequence=48, entities=7, pairs=11),
            ),
            self._compare(
                wrapper,
                destination,
                factory.build(batch=3, sequence=64, entities=6, pairs=10),
            ),
        ]
        ner_deviation = max(check["ner"] for check in checks)
        relation_deviation = max(check["relation"] for check in checks)
        relative = max(check["relative"] for check in checks)
        decisions_match = all(check["decisions"] for check in checks)

        tolerance = self._config.parity_tolerance
        if relative > tolerance or not decisions_match:
            raise ExportVerificationError(
                f"Exported graph deviates from PyTorch by {relative:.3e} "
                f"relative to the logit range (tolerance {tolerance:.3e}), "
                f"decisions match: {decisions_match}."
            )

        return ExportReport(
            path=destination,
            exporter=exporter,
            opset_version=self._config.opset_version,
            max_ner_deviation=ner_deviation,
            max_relation_deviation=relation_deviation,
            max_relative_deviation=relative,
            decisions_match=decisions_match,
            dynamic_shapes_verified=True,
            size_bytes=destination.stat().st_size,
        )

    def _write_graph(
        self,
        wrapper: OnnxExportWrapper,
        sample: dict[str, torch.Tensor],
        destination: Path,
    ) -> str:
        """Serialise the graph, preferring the dynamo exporter.

        The TorchScript exporter is retained as a fallback because the dynamo
        path can reject models whose dependencies emit unsupported constructs,
        and a working graph matters more than which backend produced it.

        Args:
            wrapper: Tuple returning wrapper around the model.
            sample: Example inputs driving the trace.
            destination: Path of the ``.onnx`` file to write.

        Returns:
            The name of the backend that produced the graph.
        """
        arguments = tuple(sample[name] for name in INPUT_NAMES)
        try:
            torch.onnx.export(
                wrapper,
                arguments,
                str(destination),
                input_names=list(INPUT_NAMES),
                output_names=list(OUTPUT_NAMES),
                opset_version=self._config.opset_version,
                dynamic_shapes=self._dynamo_dynamic_shapes(),
                dynamo=True,
                external_data=False,
                optimize=True,
            )
            return "dynamo"
        except Exception:
            torch.onnx.export(
                wrapper,
                arguments,
                str(destination),
                input_names=list(INPUT_NAMES),
                output_names=list(OUTPUT_NAMES),
                opset_version=self._config.opset_version,
                dynamic_axes=DYNAMIC_AXES,
                dynamo=False,
                do_constant_folding=True,
            )
            return "torchscript"

    def _dynamo_dynamic_shapes(self) -> dict[str, dict[int, str]]:
        """Return the dynamic axis specification for the dynamo exporter.

        Returns:
            Mapping of input name to dynamic axis names.
        """
        return {name: DYNAMIC_AXES[name] for name in INPUT_NAMES}

    def _compare(
        self,
        wrapper: OnnxExportWrapper,
        destination: Path,
        sample: dict[str, torch.Tensor],
    ) -> dict[str, float | bool]:
        """Measure how far the exported graph departs from PyTorch.

        Absolute logit deviation is reported, but the gate is applied to the
        deviation relative to the logit range, and to whether both
        implementations still choose the same class. A deeper encoder legitimately
        accumulates more floating point difference than a shallow one, so an
        absolute threshold calibrated on one backbone rejects another for no real
        reason. What must not change is the decision.

        Args:
            wrapper: Tuple returning wrapper around the model.
            destination: Path of the exported graph.
            sample: Inputs fed to both implementations.

        Returns:
            Absolute deviations, the relative deviation and decision agreement.
        """
        import onnxruntime

        with torch.no_grad():
            expected_ner, expected_relation = wrapper(**sample)

        session = onnxruntime.InferenceSession(
            str(destination), providers=["CPUExecutionProvider"]
        )
        feeds = {name: sample[name].numpy() for name in INPUT_NAMES}
        actual_ner, actual_relation = session.run(list(OUTPUT_NAMES), feeds)

        reference_ner = expected_ner.numpy()
        reference_relation = expected_relation.numpy()
        ner_deviation = float(np.abs(reference_ner - actual_ner).max())
        relation_deviation = float(np.abs(reference_relation - actual_relation).max())
        scale = max(
            float(np.abs(reference_ner).max()),
            float(np.abs(reference_relation).max()),
            1e-6,
        )
        return {
            "ner": ner_deviation,
            "relation": relation_deviation,
            "relative": max(ner_deviation, relation_deviation) / scale,
            "decisions": bool(
                (reference_ner.argmax(-1) == actual_ner.argmax(-1)).all()
                and (
                    (reference_relation > reference_relation[..., :1])
                    == (actual_relation > actual_relation[..., :1])
                ).all()
            ),
        }
