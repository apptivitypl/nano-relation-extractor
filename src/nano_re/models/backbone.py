"""Shared transformer encoder.

Wrapping the pretrained encoder keeps the multi-task model independent of the
concrete transformers class and of the keyword arguments needed to load it.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class EncoderBackbone(nn.Module):
    """Produces contextual token representations shared by both task heads.

    Args:
        encoder: A pretrained transformers encoder returning
            ``last_hidden_state``.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @classmethod
    def from_pretrained(
        cls, name_or_path: str, attn_implementation: str = "eager"
    ) -> "EncoderBackbone":
        """Load a pretrained encoder from the Hub or a local directory.

        The precision is pinned to float32 because checkpoints are published in
        varying dtypes and mixed precision must be a deliberate training choice,
        not an accident of how the weights were serialised. The eager attention
        implementation is the default because fused kernels do not trace into a
        portable ONNX graph.

        Args:
            name_or_path: Hub identifier or local path.
            attn_implementation: Attention kernel requested from transformers.

        Returns:
            The wrapped encoder.
        """
        encoder = AutoModel.from_pretrained(
            name_or_path,
            add_pooling_layer=False,
            attn_implementation=attn_implementation,
            dtype=torch.float32,
        )
        return cls(encoder)

    @classmethod
    def from_config(cls, name_or_path: str) -> "EncoderBackbone":
        """Instantiate an untrained encoder matching a pretrained config.

        Used when restoring a checkpoint, where the weights arrive from the
        checkpoint rather than from the Hub.

        Args:
            name_or_path: Hub identifier or local path of the configuration.

        Returns:
            The wrapped, randomly initialised encoder.
        """
        config = AutoConfig.from_pretrained(name_or_path)
        encoder = AutoModel.from_config(
            config, add_pooling_layer=False, attn_implementation="eager"
        )
        return cls(encoder.to(torch.float32))

    @property
    def hidden_size(self) -> int:
        """Width of the encoder's hidden representations."""
        return int(self.encoder.config.hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode a padded batch of sub-word sequences.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.

        Returns:
            Contextual token representations, shape ``[B, S, H]``.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
