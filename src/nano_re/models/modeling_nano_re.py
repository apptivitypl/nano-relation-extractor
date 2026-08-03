"""The multi-task NER and relation extraction model.

The model is a pure composition of four injected components. It contains no
construction logic, no persistence logic and no knowledge of where its weights
come from, so replacing the backbone or either head requires no edit here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .backbone import EncoderBackbone
from .heads import EntityPooler, PairwiseRelationHead, TokenClassificationHead
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
    """

    backbone_name: str
    hidden_size: int
    num_bio_labels: int
    num_relation_labels: int
    pair_hidden_size: int
    dropout: float

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
        )


class NanoREModel(nn.Module):
    """Shared encoder with a token classification head and a relation head.

    Args:
        backbone: Shared contextual encoder.
        ner_head: Token classification head.
        entity_pooler: Layer pooling mention tokens into entity vectors.
        relation_head: Pairwise relation classifier.
        architecture: Serialisable description used when persisting the model.
    """

    def __init__(
        self,
        backbone: EncoderBackbone,
        ner_head: TokenClassificationHead,
        entity_pooler: EntityPooler,
        relation_head: PairwiseRelationHead,
        architecture: NanoREArchitecture,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ner_head = ner_head
        self.entity_pooler = entity_pooler
        self.relation_head = relation_head
        self._architecture = architecture

    @property
    def architecture(self) -> NanoREArchitecture:
        """Description required to rebuild this model from a checkpoint."""
        return self._architecture

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
        hidden_states = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        ner_logits = self.ner_head(hidden_states)
        entity_representations = self.entity_pooler(hidden_states, mention_mask)
        relation_logits = self.relation_head(entity_representations, pair_index)
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
