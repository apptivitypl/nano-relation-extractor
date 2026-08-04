"""The multi-task NER and relation extraction model.

The model is a pure composition of four injected components. It contains no
construction logic, no persistence logic and no knowledge of where its weights
come from, so replacing the backbone or either head requires no edit here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from .backbone import EncoderBackbone
from .heads import (
    EntityPooler,
    LocalizedContextPooler,
    PairwiseRelationHead,
    TokenClassificationHead,
)
from .outputs import MultiTaskOutput


@dataclass(frozen=True)
class NanoREArchitecture:
    """Serialisable description of an assembled model.

    Attributes:
        backbone_name: Hub identifier of the shared encoder.
        hidden_size: Width of the encoder's hidden representations.
        num_bio_labels: Width of the token classification head.
        num_relation_labels: Width of the relation head, including ``NA``.
        pair_hidden_size: Width of the relation head's hidden layer.
        dropout: Dropout probability used in both heads.
        vocab_size: Embedding rows, which differ from the pretrained vocabulary
            once trimming has run.
        original_vocab_size: Size of the identifier lookup table, ``None`` when
            the vocabulary was never trimmed.
        localized_context: Whether the relation head receives a pair specific
            context vector built from encoder attention.
    """

    backbone_name: str
    hidden_size: int
    num_bio_labels: int
    num_relation_labels: int
    pair_hidden_size: int
    dropout: float
    vocab_size: int | None = None
    original_vocab_size: int | None = None
    localized_context: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the architecture."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NanoREArchitecture":
        """Rebuild an architecture description from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed description.
        """
        return cls(
            backbone_name=str(payload["backbone_name"]),
            hidden_size=int(payload["hidden_size"]),
            num_bio_labels=int(payload["num_bio_labels"]),
            num_relation_labels=int(payload["num_relation_labels"]),
            pair_hidden_size=int(payload["pair_hidden_size"]),
            dropout=float(payload["dropout"]),
            vocab_size=(
                int(payload["vocab_size"])
                if payload.get("vocab_size") is not None
                else None
            ),
            original_vocab_size=(
                int(payload["original_vocab_size"])
                if payload.get("original_vocab_size") is not None
                else None
            ),
            localized_context=bool(payload.get("localized_context", False)),
        )


class NanoREModel(nn.Module):
    """Shared encoder with a token classification head and a relation head.

    Args:
        backbone: Shared contextual encoder.
        ner_head: Token classification head.
        entity_pooler: Layer pooling mention tokens into entity vectors.
        relation_head: Pairwise relation classifier.
        architecture: Serialisable description used when persisting the model.
        context_pooler: Optional layer building a pair specific context vector.
    """

    def __init__(
        self,
        backbone: EncoderBackbone,
        ner_head: TokenClassificationHead,
        entity_pooler: EntityPooler,
        relation_head: PairwiseRelationHead,
        architecture: NanoREArchitecture,
        context_pooler: LocalizedContextPooler | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ner_head = ner_head
        self.entity_pooler = entity_pooler
        self.relation_head = relation_head
        self.context_pooler = context_pooler
        self._architecture = architecture

    @property
    def architecture(self) -> NanoREArchitecture:
        """Description required to rebuild this model from a checkpoint.

        The vocabulary fields are read from the backbone on every access rather
        than captured at construction, because trimming replaces the embedding
        table after the model is assembled. A description recorded once would
        still name the pretrained vocabulary and the checkpoint would refuse to
        load.
        """
        return replace(
            self._architecture,
            vocab_size=int(self.backbone.encoder.config.vocab_size),
            original_vocab_size=self.backbone.original_vocab_size,
            localized_context=self.context_pooler is not None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> MultiTaskOutput:
        """Run both tasks over a padded batch.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Row-normalised entity pooling weights, ``[B, E, S]``.
            pair_index: Head and tail entity rows per pair, shape ``[B, P, 2]``.

        Returns:
            Logits for both tasks together with the pooled entity vectors.
        """
        if self.context_pooler is None:
            hidden_states = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            )
            context = None
        else:
            hidden_states, attention = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attention=True,
            )
            context = self.context_pooler(
                hidden_states, attention, mention_mask, pair_index
            )
        ner_logits = self.ner_head(hidden_states)
        entity_representations = self.entity_pooler(hidden_states, mention_mask)
        relation_logits = self.relation_head(
            entity_representations, pair_index, context=context
        )
        return MultiTaskOutput(
            ner_logits=ner_logits,
            relation_logits=relation_logits,
            entity_representations=entity_representations,
        )

    def parameter_groups(
        self, weight_decay: float, encoder_lr: float, head_lr: float
    ) -> list[dict[str, object]]:
        """Split parameters into optimiser groups.

        The pretrained encoder and the randomly initialised heads are given
        separate learning rates, and normalisation and bias tensors are excluded
        from weight decay.

        Args:
            weight_decay: Decay applied to eligible parameters.
            encoder_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for head parameters.

        Returns:
            Parameter group dictionaries accepted by ``torch.optim.AdamW``.
        """
        no_decay_markers = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
            (True, True): [],
            (True, False): [],
            (False, True): [],
            (False, False): [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            is_backbone = name.startswith("backbone.")
            applies_decay = not any(marker in name for marker in no_decay_markers)
            groups[(is_backbone, applies_decay)].append(parameter)

        result: list[dict[str, object]] = []
        for (is_backbone, applies_decay), parameters in groups.items():
            if not parameters:
                continue
            result.append(
                {
                    "params": parameters,
                    "lr": encoder_lr if is_backbone else head_lr,
                    "weight_decay": weight_decay if applies_decay else 0.0,
                }
            )
        return result


class OnnxExportWrapper(nn.Module):
    """Adapts :class:`NanoREModel` to the tuple output ONNX export expects.

    Args:
        model: The multi-task model to wrap.
    """

    def __init__(self, model: NanoREModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both task logits as a plain tuple.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Entity pooling weights, shape ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            A tuple of NER logits and relation logits.
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mention_mask=mention_mask,
            pair_index=pair_index,
        )
        return output.ner_logits, output.relation_logits


class DataParallelAdapter(nn.Module):
    """Runs the model across several GPUs on one machine.

    ``DataParallel`` splits a batch across devices and gathers the results, but
    it can only gather tensors and the standard containers. This model returns a
    dataclass, which would fail there, so the wrapped module returns a tuple and
    the dataclass is rebuilt afterwards.

    Data parallelism replicates the model on every forward pass and reduces
    gradients on one device, so two cards do not halve the time. It is here
    because a second idle card is worse than a partially used one, not because
    it is the fastest way to use several GPUs; that is distributed training,
    which needs a process launcher a notebook cannot provide.

    Args:
        model: The multi-task model to parallelise.
        device_ids: Devices to spread the batch across.
    """

    def __init__(self, model: NanoREModel, device_ids: list[int]) -> None:
        super().__init__()
        self.module = model
        self._parallel = nn.DataParallel(
            _TupleOutputWrapper(model), device_ids=device_ids
        )
        self._device_ids = list(device_ids)

    @property
    def architecture(self) -> NanoREArchitecture:
        """Description required to rebuild the wrapped model."""
        return self.module.architecture

    @property
    def backbone(self) -> EncoderBackbone:
        """Shared encoder of the wrapped model."""
        return self.module.backbone

    @property
    def device_ids(self) -> list[int]:
        """Devices the batch is spread across."""
        return list(self._device_ids)

    def parameter_groups(
        self, weight_decay: float, encoder_lr: float, head_lr: float
    ) -> list[dict[str, object]]:
        """Delegate optimiser grouping to the wrapped model.

        Args:
            weight_decay: Decay applied to eligible parameters.
            encoder_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for head parameters.

        Returns:
            Parameter group dictionaries accepted by ``torch.optim.AdamW``.
        """
        return self.module.parameter_groups(weight_decay, encoder_lr, head_lr)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> MultiTaskOutput:
        """Run the batch across every device and rebuild the output.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Entity pooling weights, shape ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            The combined logits for both tasks.
        """
        ner_logits, relation_logits, entities = self._parallel(
            input_ids, attention_mask, mention_mask, pair_index
        )
        return MultiTaskOutput(
            ner_logits=ner_logits,
            relation_logits=relation_logits,
            entity_representations=entities,
        )


class _TupleOutputWrapper(nn.Module):
    """Presents the model's dataclass output as a gatherable tuple.

    Args:
        model: The multi-task model to wrap.
    """

    def __init__(self, model: NanoREModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the model's three tensors as a plain tuple.

        Args:
            input_ids: Sub-word identifiers, shape ``[B, S]``.
            attention_mask: Padding mask, shape ``[B, S]``.
            mention_mask: Entity pooling weights, shape ``[B, E, S]``.
            pair_index: Head and tail entity rows, shape ``[B, P, 2]``.

        Returns:
            NER logits, relation logits and entity representations.
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mention_mask=mention_mask,
            pair_index=pair_index,
        )
        return (
            output.ner_logits,
            output.relation_logits,
            output.entity_representations,
        )
