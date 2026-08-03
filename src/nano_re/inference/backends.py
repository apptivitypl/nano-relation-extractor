"""Interchangeable execution backends for inference.

The extractor is written against one small interface, so the same extraction
logic runs on the PyTorch checkpoint or on either ONNX graph. Choosing a backend
is a deployment decision, not a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from ..export.runtime import OnnxInferenceSession, build_feeds
from ..models import NanoREModel, NanoREModelFactory


@runtime_checkable
class InferenceBackend(Protocol):
    """Runs the multi-task model over a prepared batch."""

    @property
    def name(self) -> str:
        """Identifier reported to the user."""
        ...

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the model.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Entity pooling weights, shape ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            The NER logits and the relation logits.
        """
        ...


class TorchBackend:
    """Runs the PyTorch checkpoint on CPU.

    Args:
        model: Trained model, switched to evaluation mode.
    """

    def __init__(self, model: NanoREModel) -> None:
        self._model = model.eval()

    @classmethod
    def from_bundle(cls, directory: Path) -> "TorchBackend":
        """Load the checkpoint from a bundle directory.

        Args:
            directory: Bundle produced by the packaging stage.

        Returns:
            The configured backend.
        """
        return cls(NanoREModelFactory().load(directory))

    @property
    def name(self) -> str:
        """Identifier reported to the user."""
        return "pytorch"

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the checkpoint.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Entity pooling weights, shape ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            The NER logits and the relation logits.
        """
        output = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mention_mask=mention_mask,
            pair_index=pair_index,
        )
        return output.ner_logits, output.relation_logits


class OnnxBackend:
    """Runs an ONNX graph through ONNX Runtime on CPU.

    Args:
        session: Loaded ONNX Runtime session.
        name: Identifier reported to the user.
    """

    def __init__(self, session: OnnxInferenceSession, name: str = "onnx") -> None:
        self._session = session
        self._name = name

    @classmethod
    def from_bundle(
        cls,
        directory: Path,
        filename: str = "model_int8.onnx",
        intra_op_num_threads: int = 0,
    ) -> "OnnxBackend":
        """Load a graph from a bundle directory.

        Args:
            directory: Bundle produced by the packaging stage.
            filename: Graph to load.
            intra_op_num_threads: CPU threads per operator, ``0`` for automatic.

        Returns:
            The configured backend.

        Raises:
            FileNotFoundError: If the graph is absent from the bundle.
        """
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run the export stage, or choose a "
                "backend whose artifact is present."
            )
        label = "onnx-int8" if "int8" in filename else "onnx-fp32"
        return cls(
            OnnxInferenceSession(path, intra_op_num_threads=intra_op_num_threads),
            name=label,
        )

    @property
    def name(self) -> str:
        """Identifier reported to the user."""
        return self._name

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the graph.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Entity pooling weights, shape ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            The NER logits and the relation logits.
        """
        feeds = build_feeds(input_ids, attention_mask, mention_mask, pair_index)
        ner_logits, relation_logits = self._session.run(feeds)
        return torch.from_numpy(ner_logits), torch.from_numpy(relation_logits)
