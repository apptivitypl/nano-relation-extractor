"""Assembly and persistence of :class:`NanoREModel` instances.

Construction and serialisation live outside the model so the model itself stays
a pure composition of layers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..config import ModelConfig
from ..schema import LabelSchema
from .backbone import EncoderBackbone
from .heads import EntityPooler, PairwiseRelationHead, TokenClassificationHead
from .modeling_nano_re import NanoREArchitecture, NanoREModel

ARCHITECTURE_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"


class NanoREModelFactory:
    """Builds models from configuration or from a saved checkpoint."""

    def build(self, model_config: ModelConfig, schema: LabelSchema) -> NanoREModel:
        """Assemble a model with a pretrained backbone and fresh heads.

        Args:
            model_config: Architecture settings.
            schema: Label vocabularies determining both head widths.

        Returns:
            The assembled model.
        """
        backbone = EncoderBackbone.from_pretrained(model_config.backbone_name)
        architecture = NanoREArchitecture(
            backbone_name=model_config.backbone_name,
            hidden_size=backbone.hidden_size,
            num_bio_labels=schema.num_bio_labels,
            num_relation_labels=schema.num_relation_labels,
            pair_hidden_size=model_config.pair_hidden_size,
            dropout=model_config.dropout,
            vocab_size=int(backbone.encoder.config.vocab_size),
            original_vocab_size=backbone.original_vocab_size,
        )
        return self._assemble(backbone, architecture)

    def load(self, directory: Path) -> NanoREModel:
        """Rebuild a model from a directory written by :meth:`save`.

        Args:
            directory: Directory holding the architecture and weight files.

        Returns:
            The restored model in evaluation mode.

        Raises:
            FileNotFoundError: If either artifact is missing.
        """
        architecture_path = directory / ARCHITECTURE_FILENAME
        weights_path = directory / WEIGHTS_FILENAME
        if not architecture_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Expected {ARCHITECTURE_FILENAME} and {WEIGHTS_FILENAME} in "
                f"{directory}. Run the training stage first."
            )
        payload = json.loads(architecture_path.read_text(encoding="utf-8"))
        architecture = NanoREArchitecture.from_dict(payload)
        backbone = EncoderBackbone.from_config(
            architecture.backbone_name,
            vocab_size=architecture.vocab_size,
            original_vocab_size=architecture.original_vocab_size,
        )
        model = self._assemble(backbone, architecture)
        model.load_state_dict(load_file(str(weights_path)))
        model.eval()
        return model

    def save(self, model: NanoREModel, directory: Path) -> Path:
        """Write a model's architecture and weights to a directory.

        Args:
            model: Trained model to persist.
            directory: Destination directory, created when missing.

        Returns:
            The directory that was written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARCHITECTURE_FILENAME).write_text(
            json.dumps(model.architecture.to_dict(), indent=2), encoding="utf-8"
        )
        state_dict = {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        }
        save_file(state_dict, str(directory / WEIGHTS_FILENAME))
        return directory

    def _assemble(
        self, backbone: EncoderBackbone, architecture: NanoREArchitecture
    ) -> NanoREModel:
        """Wire a backbone and freshly built heads into a model.

        Args:
            backbone: Shared encoder.
            architecture: Description determining head widths.

        Returns:
            The assembled model.
        """
        return NanoREModel(
            backbone=backbone,
            ner_head=TokenClassificationHead(
                hidden_size=architecture.hidden_size,
                num_labels=architecture.num_bio_labels,
                dropout=architecture.dropout,
            ),
            entity_pooler=EntityPooler(),
            relation_head=PairwiseRelationHead(
                hidden_size=architecture.hidden_size,
                num_relations=architecture.num_relation_labels,
                pair_hidden_size=architecture.pair_hidden_size,
                dropout=architecture.dropout,
            ),
            architecture=architecture,
        )


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of parameters in a module.

    Args:
        model: Module to inspect.

    Returns:
        Parameter count across all submodules.
    """
    return sum(parameter.numel() for parameter in model.parameters())
