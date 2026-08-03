"""Command line entry point.

Each subcommand is a thin wrapper around one pipeline stage, or over the
extractor for the ``extract`` command. Configuration comes from the environment,
and the few options that change between runs rather than between deployments are
exposed as flags.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .inference import ExtractionConsole, ExtractionSettings, RelationExtractor
from .pipeline import Pipeline

STAGES = (
    "prepare",
    "train",
    "export",
    "benchmark",
    "package",
    "all",
    "extract",
)
BACKENDS = ("onnx-int8", "onnx-fp32", "pytorch")


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
        help=(
            "Stage to run. 'all' runs the whole pipeline; 'extract' runs a "
            "trained bundle over your own text."
        ),
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
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Initialise training from an existing bundle instead of the "
            "pretrained backbone. Chains the two-stage recipe: pretrain on "
            "train_distant, then fine-tune on train_annotated from that "
            "checkpoint."
        ),
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

    extraction = parser.add_argument_group("extract")
    extraction.add_argument(
        "--text",
        default=None,
        help="Text to extract from. Without it, input is read from a pipe or "
        "typed interactively.",
    )
    extraction.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Read the input text from a file.",
    )
    extraction.add_argument(
        "--backend",
        choices=BACKENDS,
        default="onnx-int8",
        help="Which artifact to run. Defaults to the quantised ONNX graph.",
    )
    extraction.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human readable report.",
    )
    extraction.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Drop relations scoring below this confidence.",
    )
    extraction.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Report at most this many relations. 0 means no cap.",
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
    if arguments.init_from is not None:
        training = replace(training, init_from=arguments.init_from)
    if arguments.drop_fp32_graph:
        packaging = replace(packaging, keep_fp32_graph=False)

    return config.with_overrides(
        data=data, training=training, packaging=packaging
    )


def resolve_input_text(arguments: argparse.Namespace) -> str | None:
    """Determine the text to extract from, if one was supplied.

    Args:
        arguments: Parsed command line arguments.

    Returns:
        The input text, or ``None`` when the session should be interactive.

    Raises:
        FileNotFoundError: If ``--file`` names a path that does not exist.
    """
    if arguments.text is not None:
        return arguments.text
    if arguments.file is not None:
        if not arguments.file.exists():
            raise FileNotFoundError(f"{arguments.file} does not exist.")
        return arguments.file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


def run_extract(config: PipelineConfig, arguments: argparse.Namespace) -> int:
    """Run the extractor over supplied text or an interactive session.

    Args:
        config: Resolved configuration.
        arguments: Parsed command line arguments.

    Returns:
        A process exit code.
    """
    text = resolve_input_text(arguments)
    extractor = RelationExtractor.from_bundle(
        config.artifacts_dir,
        backend=arguments.backend,
        config=config,
        settings=ExtractionSettings(
            max_sequence_length=config.data.max_sequence_length,
            min_confidence=arguments.min_confidence,
            top_k=arguments.top_k,
        ),
    )
    console = ExtractionConsole(extractor, as_json=arguments.json)
    if text is None:
        console.run_interactive()
    else:
        console.run_once(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the requested stage.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    arguments = build_parser().parse_args(argv)
    config = apply_overrides(PipelineConfig.from_env(), arguments)

    try:
        if arguments.stage == "extract":
            return run_extract(config, arguments)

        pipeline = Pipeline(config)
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
