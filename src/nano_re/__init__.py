"""A lightweight multi-task NER and relation extraction pipeline.

The package is organised so that each stage of the pipeline can be used on its
own: data acquisition and encoding, model assembly, multi-task training, ONNX
export with INT8 quantisation, and assembly of a self-contained local bundle.

Nothing is ever uploaded and no credential is required. The Hugging Face Hub is
used only to download the public multilingual corpora and the public
pretrained encoder.
"""

from .config import (
    DataConfig,
    ExportConfig,
    ModelConfig,
    PackagingConfig,
    PipelineConfig,
    TrainingConfig,
)
from .pipeline import Pipeline
from .schema import LabelSchema

__all__ = [
    "DataConfig",
    "ExportConfig",
    "LabelSchema",
    "ModelConfig",
    "PackagingConfig",
    "Pipeline",
    "PipelineConfig",
    "TrainingConfig",
]

__version__ = "0.1.0"
