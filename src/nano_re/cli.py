"""Command line entry point.

Each subcommand is a thin wrapper around one pipeline stage. Configuration comes
from the environment, and the few options that change between runs rather than
between deployments are exposed as flags.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .pipeline import Pipeline

STAGES = ("prepare", "train", "export", "benchmark", "package", "all")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="nano-re",
        description=(
            "Train, quantise and package a multi-task NER and relation "
            "extraction model into a local bundle."
        ),
    )
    parser.add_argument(
        "stage",
        choices=STAGES,
        help="Pipeline stage to run. 'all' runs every stage in order.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of documents per split. Useful for smoke tests.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the number of epochs."
    )
    parser.add_argument(
        "--train-split", default=None, help="Override the training split name."
    )
    parser.add_argument(
        "--eval-split", default=None, help="Override the evaluation split name."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the artifact directory.",
    )
    parser.add_argument(
        "--relation-loss",
        choices=("adaptive_threshold", "bce"),
        default=None,
        help="Override the relation objective.",
    )
    parser.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip scoring both graphs during the benchmark stage.",
    )
    parser.add_argument(
        "--drop-fp32-graph",
        action="store_true",
        help=(
            "Delete the float32 ONNX graph when packaging. Saves roughly four "
            "times the INT8 size on disk, but the benchmark can no longer "
            "compare against it."
        ),
    )
    return parser


def apply_overrides(
    config: PipelineConfig, arguments: argparse.Namespace
) -> PipelineConfig:
    """Apply command line overrides on top of the environment configuration.

    Args:
        config: Configuration built from the environment.
        arguments: Parsed command line arguments.

    Returns:
        The overridden configuration.
    """
    data = config.data
    training = config.training
    packaging = config.packaging

    if arguments.limit is not None:
        data = replace(data, limit=arguments.limit)
    if arguments.train_split is not None:
        data = replace(data, train_split=arguments.train_split)
    if arguments.eval_split is not None:
        data = replace(data, eval_split=arguments.eval_split)
    if arguments.epochs is not None:
        training = replace(training, epochs=arguments.epochs)
    if arguments.relation_loss is not None:
        training = replace(training, relation_loss=arguments.relation_loss)
    if arguments.output_dir is not None:
        training = replace(training, output_dir=arguments.output_dir)
    if arguments.drop_fp32_graph:
        packaging = replace(packaging, keep_fp32_graph=False)

    return config.with_overrides(
        data=data, training=training, packaging=packaging
    )


def main(argv: list[str] | None = None) -> int:
    """Run the requested pipeline stage.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    arguments = build_parser().parse_args(argv)
    config = apply_overrides(PipelineConfig.from_env(), arguments)
    pipeline = Pipeline(config)

    try:
        if arguments.stage == "prepare":
            pipeline.prepare()
        elif arguments.stage == "train":
            pipeline.train()
        elif arguments.stage == "export":
            pipeline.export()
        elif arguments.stage == "benchmark":
            pipeline.benchmark(measure_accuracy=not arguments.no_accuracy)
        elif arguments.stage == "package":
            pipeline.package()
        else:
            pipeline.run_all()
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
