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

    When the embedding table has been trimmed to a subset of the pretrained
    vocabulary, the backbone also carries a lookup table translating original
    token identifiers into compacted rows. Keeping that translation inside the
    model means the tokenizer, and therefore every consumer of the bundle, works
    with the identifiers it always did.

    Args:
        encoder: A pretrained transformers encoder returning
            ``last_hidden_state``.
        token_remap: Optional ``[original_vocab_size]`` lookup table.
    """

    def __init__(
        self, encoder: nn.Module, token_remap: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("token_remap", token_remap, persistent=True)

    def attach_token_remap(self, token_remap: torch.Tensor) -> None:
        """Install a token identifier lookup table.

        Args:
            token_remap: A ``[original_vocab_size]`` long tensor.
        """
        self.token_remap = token_remap

    @property
    def original_vocab_size(self) -> int | None:
        """Size of the pretrained vocabulary, when the table was trimmed."""
        if self.token_remap is None:
            return None
        return int(self.token_remap.shape[0])

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
    def from_config(
        cls,
        name_or_path: str,
        vocab_size: int | None = None,
        original_vocab_size: int | None = None,
    ) -> "EncoderBackbone":
        """Instantiate an untrained encoder matching a pretrained config.

        Used when restoring a checkpoint, where the weights arrive from the
        checkpoint rather than from the Hub. A checkpoint whose vocabulary was
        trimmed states both sizes, so the rebuilt module has exactly the shapes
        the saved tensors expect.

        Args:
            name_or_path: Hub identifier or local path of the configuration.
            vocab_size: Embedding rows to allocate, when trimmed.
            original_vocab_size: Size of the lookup table, when trimmed.

        Returns:
            The wrapped, randomly initialised encoder.
        """
        config = AutoConfig.from_pretrained(name_or_path)
        if vocab_size is not None:
            config.vocab_size = vocab_size
        encoder = AutoModel.from_config(
            config, add_pooling_layer=False, attn_implementation="eager"
        )
        remap = (
            torch.zeros(original_vocab_size, dtype=torch.long)
            if original_vocab_size is not None
            else None
        )
        return cls(encoder.to(torch.float32), token_remap=remap)

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
        if self.token_remap is not None:
            input_ids = self.token_remap[input_ids]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
