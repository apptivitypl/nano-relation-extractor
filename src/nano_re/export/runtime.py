"""ONNX Runtime inference wrapper.

The adapter presents an ONNX session behind the same call signature and return
type as :class:`~nano_re.models.NanoREModel`, so the existing evaluator scores a
quantised graph without a second evaluation loop being written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from ..models import MultiTaskOutput
from .onnx_exporter import INPUT_NAMES, OUTPUT_NAMES


class OnnxInferenceSession:
    """Loads an ONNX graph and runs it on CPU.

    Args:
        model_path: Path of the ``.onnx`` file.
        intra_op_num_threads: CPU threads per operator. ``0`` lets ONNX Runtime
            choose based on the host.
    """

    def __init__(self, model_path: Path, intra_op_num_threads: int = 0) -> None:
        options = onnxruntime.SessionOptions()
        if intra_op_num_threads > 0:
            options.intra_op_num_threads = intra_op_num_threads
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._path = model_path
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    @property
    def path(self) -> Path:
        """Location of the loaded graph."""
        return self._path

    @property
    def size_bytes(self) -> int:
        """Size of the loaded graph on disk."""
        return self._path.stat().st_size

    def run(self, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Execute the graph.

        Args:
            feeds: Mapping of input name to array.

        Returns:
            The NER logits and the relation logits.
        """
        outputs = self._session.run(list(OUTPUT_NAMES), feeds)
        return outputs[0], outputs[1]


class OnnxModelAdapter(nn.Module):
    """Exposes an ONNX session through the PyTorch model interface.

    Args:
        session: Loaded ONNX Runtime session.
    """

    def __init__(self, session: OnnxInferenceSession) -> None:
        super().__init__()
        self._session = session

    @property
    def session(self) -> OnnxInferenceSession:
        """The wrapped ONNX Runtime session."""
        return self._session

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> MultiTaskOutput:
        """Run the graph and return logits as tensors.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Entity pooling weights, shape ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            Both task logits. Entity representations are not exposed by the
            graph and are returned empty.
        """
        feeds = build_feeds(input_ids, attention_mask, mention_mask, pair_index)
        ner_logits, relation_logits = self._session.run(feeds)
        return MultiTaskOutput(
            ner_logits=torch.from_numpy(ner_logits),
            relation_logits=torch.from_numpy(relation_logits),
            entity_representations=torch.empty(0),
        )


def build_feeds(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mention_mask: torch.Tensor,
    pair_index: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Convert model inputs into ONNX Runtime feeds.

    Args:
        input_ids: Sub-word identifiers, shape ``[B, S]``.
        attention_mask: Padding mask, shape ``[B, S]``.
        mention_mask: Entity pooling weights, shape ``[B, E, S]``.
        pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

    Returns:
        Mapping of graph input name to a contiguous NumPy array.
    """
    tensors = {
        "input_ids": input_ids.to(torch.int64),
        "attention_mask": attention_mask.to(torch.int64),
        "mention_mask": mention_mask.to(torch.float32),
        "pair_index": pair_index.to(torch.int64),
    }
    return {
        name: np.ascontiguousarray(tensors[name].detach().cpu().numpy())
        for name in INPUT_NAMES
    }
