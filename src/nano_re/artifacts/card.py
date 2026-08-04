"""Automated model card generation.

Every number in the card is interpolated from the reports produced by the
training, export and benchmark stages. Nothing is written by hand, so the card
cannot drift away from the artifacts it sits beside.

The card is plain Markdown with no Hub metadata block: it documents a local
bundle rather than a remote repository.
"""

from __future__ import annotations

from pathlib import Path

from ..config import PackagingConfig
from ..export.benchmark import BenchmarkReport
from ..export.quantizer import QuantizationReport
from ..schema import LabelSchema
from ..training.trainer import TrainingReport


class ModelCardBuilder:
    """Renders a Markdown model card from pipeline reports.

    Args:
        config: Packaging settings supplying the model name and license.
        corpora: Corpora the model was trained on.
    """

    def __init__(
        self,
        config: PackagingConfig,
        corpora: tuple[str, ...] = ("sredfm", "multinerd"),
    ) -> None:
        self._config = config
        self._corpora = tuple(corpora)

    def build(
        self,
        schema: LabelSchema,
        training: TrainingReport,
        benchmark: BenchmarkReport | None = None,
        quantization: QuantizationReport | None = None,
        parameter_count: int | None = None,
    ) -> str:
        """Render the card.

        Args:
            schema: Label vocabularies describing both task outputs.
            training: Report produced by the trainer.
            benchmark: Optional CPU latency comparison.
            quantization: Optional quantisation report.
            parameter_count: Optional total parameter count.

        Returns:
            The rendered Markdown document.
        """
        self._bio_width = schema.num_bio_labels
        self._relation_width = schema.num_relation_labels
        self._int8_usable = quantization is None or quantization.is_usable
        sections = [
            self._header(training, parameter_count),
            self._contents(),
            self._architecture(schema, training),
            self._results(training),
            self._cpu_section(benchmark, quantization),
            self._usage(),
            self._training_details(training),
            self._limitations(training),
            self._citation(),
        ]
        return "\n\n".join(section for section in sections if section) + "\n"

    def write(self, card: str, path: Path) -> Path:
        """Write a rendered card to disk.

        Args:
            card: Markdown produced by :meth:`build`.
            path: Destination file path.

        Returns:
            The path that was written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(card, encoding="utf-8")
        return path

    def _header(self, training: TrainingReport, parameter_count: int | None) -> str:
        """Render the title and summary section.

        Args:
            training: Report produced by the trainer.
            parameter_count: Optional total parameter count.

        Returns:
            Markdown text.
        """
        size = (
            f" It has {parameter_count / 1e6:.1f}M parameters."
            if parameter_count
            else ""
        )
        return (
            f"# {self._config.model_name}\n\n"
            "A lightweight multi-task model that performs named entity "
            "recognition and document-level relation extraction in a single "
            f"forward pass over a shared `{training.backbone_name}` encoder."
            f"{size} An INT8 quantised ONNX graph sits alongside the PyTorch "
            "weights for CPU deployment.\n\n"
            f"License: `{self._config.license_id}`. Trained on "
            f"`{'`, `'.join(self._corpora)}`."
        )

    def _contents(self) -> str:
        """Render the bundle inventory section.

        Returns:
            Markdown text.
        """
        rows = [
            ("`model.safetensors`", "Trained PyTorch weights"),
            (
                "`config.json`",
                "Architecture description needed to rebuild the model",
            ),
            (
                "`model_int8.onnx`",
                "INT8 quantised graph for CPU inference"
                if self._int8_usable
                else "INT8 graph, **not usable**, see CPU performance",
            ),
        ]
        if self._config.keep_fp32_graph:
            rows.append(
                ("`model.onnx`", "Float32 graph, kept for benchmark comparison")
            )
        rows.extend(
            [
                ("`tokenizer.json`, `tokenizer_config.json`", "Tokenizer"),
                ("`label_schema.json`", "BIO tag and relation vocabularies"),
                (
                    "`training_report.json`",
                    "Per-epoch losses and evaluation scores",
                ),
                (
                    "`export_report.json`",
                    "Export verification and quantisation results",
                ),
                ("`benchmark.json`", "CPU latency and accuracy comparison"),
                ("`MANIFEST.json`", "File inventory with sizes"),
            ]
        )
        table = "\n".join(f"| {name} | {purpose} |" for name, purpose in rows)
        return "## Bundle contents\n\n| File | Purpose |\n| --- | --- |\n" + table

    def _architecture(self, schema: LabelSchema, training: TrainingReport) -> str:
        """Render the architecture section.

        Args:
            schema: Label vocabularies describing both task outputs.
            training: Report produced by the trainer.

        Returns:
            Markdown text.
        """
        return (
            "## Architecture\n\n"
            f"- **Shared encoder**: `{training.backbone_name}`\n"
            f"- **NER head**: linear token classifier over {schema.num_bio_labels} "
            f"BIO tags covering {', '.join(schema.entity_types)}\n"
            "- **Entity pooling**: mean over the sub-word tokens of every mention "
            "in a coreference cluster\n"
            "- **Relation head**: MLP over the concatenation of the head vector, "
            "the tail vector, their element-wise product and their absolute "
            f"difference, scoring {schema.num_relation_labels} classes "
            f"({len(schema.relation_ids)} relations plus `NA`)\n"
            f"- **Relation objective**: `{training.relation_objective}`"
        )

    def _results(self, training: TrainingReport) -> str:
        """Render the evaluation section.

        Args:
            training: Report produced by the trainer.

        Returns:
            Markdown text.
        """
        evaluation = training.best_evaluation
        if evaluation is None:
            return ""
        return (
            "## Evaluation\n\n"
            "Measured on the development split with the checkpoint from "
            f"epoch {training.best_epoch}.\n\n"
            "| Task | Precision | Recall | Micro F1 |\n"
            "| --- | --- | --- | --- |\n"
            f"| NER (span level) | {evaluation.ner.precision:.4f} | "
            f"{evaluation.ner.recall:.4f} | {evaluation.ner.f1:.4f} |\n"
            f"| Relation extraction | {evaluation.relation.precision:.4f} | "
            f"{evaluation.relation.recall:.4f} | {evaluation.relation.f1:.4f} |\n\n"
            "Relation recall is bounded above by "
            f"{evaluation.relation_recall_ceiling:.4f}: gold triples whose "
            "entities fall outside the 512 sub-word window never enter the "
            "candidate set."
        )

    def _cpu_section(
        self,
        benchmark: BenchmarkReport | None,
        quantization: QuantizationReport | None,
    ) -> str:
        """Render the CPU latency and quantisation section.

        Args:
            benchmark: Optional CPU latency comparison.
            quantization: Optional quantisation report.

        Returns:
            Markdown text.
        """
        if benchmark is None:
            return ""
        threads = "automatic" if benchmark.threads == 0 else str(benchmark.threads)
        lines = ["## CPU performance", ""]
        if quantization is not None and not quantization.is_usable:
            lines.extend([
                "> **The INT8 graph is not usable.** No quantisation "
                "configuration preserved this model's predictions "
                f"(best {quantization.agreement:.1%} agreement with float32). "
                "Deploy `model.onnx` and ignore the size figures below.",
                "",
            ])
        lines.extend([
            "Single document inference through ONNX Runtime on CPU, measured "
            f"over {benchmark.fp32.iterations} iterations across "
            f"{benchmark.documents} documents with {threads} intra-op threads.",
            "",
            "| Model | Size | Median | Mean | p95 |",
            "| --- | --- | --- | --- | --- |",
            f"| `model.onnx` (FP32) | {benchmark.fp32.size_mb:.1f} MB | "
            f"{benchmark.fp32.median_ms:.1f} ms | {benchmark.fp32.mean_ms:.1f} ms | "
            f"{benchmark.fp32.p95_ms:.1f} ms |",
            f"| `model_int8.onnx` (INT8) | {benchmark.int8.size_mb:.1f} MB | "
            f"{benchmark.int8.median_ms:.1f} ms | {benchmark.int8.mean_ms:.1f} ms | "
            f"{benchmark.int8.p95_ms:.1f} ms |",
            "",
            f"Quantisation removes {benchmark.size_reduction:.1%} of the file "
            f"size. {self._latency_verdict(benchmark)}",
        ])
        if quantization is not None:
            lines.append("")
            detail = (
                f"Quantisation used the `{quantization.recipe}` configuration, "
                f"applied to `{'`, `'.join(quantization.quantized_op_types)}`."
            )
            if quantization.agreement is not None:
                detail += (
                    f" Its predictions agree with float32 on "
                    f"{quantization.agreement:.1%} of decisions."
                )
            if quantization.rejected:
                discarded = ", ".join(
                    f"`{name}` ({score:.1%})" for name, score in quantization.rejected
                )
                detail += (
                    f" Configurations tried and rejected: {discarded}. Dynamic "
                    "quantisation pairs unsigned activations with signed weights, "
                    "and on x86 without VNNI that product saturates, so the "
                    "smallest configuration is not always the one that survives."
                )
            lines.append(detail)
        delta = benchmark.relation_f1_delta
        ner_delta = benchmark.ner_f1_delta
        if delta is not None and ner_delta is not None:
            lines.extend(
                [
                    "",
                    f"Quantisation changes NER F1 by {ner_delta:+.4f} and "
                    f"relation F1 by {delta:+.4f}.",
                ]
            )
        return "\n".join(lines)

    def _latency_verdict(self, benchmark: BenchmarkReport) -> str:
        """Describe the measured latency effect in the direction it occurred.

        Dynamic INT8 is a reliable size win but only a conditional speed win:
        it depends on the host having integer dot-product kernels. Stating the
        measured direction keeps the card honest on hardware where it does not.

        Args:
            benchmark: CPU latency comparison.

        Returns:
            A sentence describing the latency effect.
        """
        speedup = benchmark.speedup
        if speedup >= 1.05:
            return (
                f"Median latency improves by a factor of {speedup:.2f} "
                f"({benchmark.fp32.median_ms:.1f} ms to "
                f"{benchmark.int8.median_ms:.1f} ms per page)."
            )
        if speedup >= 0.95:
            return (
                "Median latency is unchanged within measurement noise "
                f"({benchmark.fp32.median_ms:.1f} ms against "
                f"{benchmark.int8.median_ms:.1f} ms per page). Dynamic INT8 "
                "accelerates inference only where the host provides integer "
                "dot-product kernels, so the size reduction is the portable "
                "benefit and the speed benefit is hardware dependent."
            )
        return (
            f"Median latency is {1.0 / speedup:.2f} times higher than float32 "
            f"({benchmark.fp32.median_ms:.1f} ms against "
            f"{benchmark.int8.median_ms:.1f} ms per page) on the benchmark host. "
            "Dynamic INT8 pays off on hosts with integer dot-product "
            "acceleration; without it, dequantisation overhead dominates and "
            "the reduction in file size is the benefit that remains."
        )

    def _usage(self) -> str:
        """Render the usage section.

        Returns:
            Markdown text.
        """
        return (
            "## Usage\n\n"
            "### Command line\n\n"
            "```bash\n"
            "uv run nano-re extract\n"
            'uv run nano-re extract --text "Skai TV is a Greek network."\n'
            "uv run nano-re extract --file article.txt --json\n"
            "```\n\n"
            "With no input argument the command starts an interactive session: "
            "paste text, press Enter on an empty line, read the extraction. It "
            "decodes the BIO tags, clusters the mentions into entities by "
            "surface form, and scores every ordered pair for relations.\n\n"
            "### ONNX graph\n\n"
            "The graph takes four inputs and returns both task outputs. "
            "All four axes are dynamic.\n\n"
            "| Tensor | Shape | Description |\n"
            "| --- | --- | --- |\n"
            "| `input_ids` | `[B, S]` int64 | Sub-word identifiers |\n"
            "| `attention_mask` | `[B, S]` int64 | Padding mask |\n"
            "| `mention_mask` | `[B, E, S]` float32 | Row-normalised pooling "
            "weights, one row per entity |\n"
            "| `pair_index` | `[B, P, 2]` int64 | Head and tail entity rows per "
            "candidate pair |\n"
            f"| `ner_logits` | `[B, S, {self._bio_width}]` float32 | BIO scores |\n"
            f"| `relation_logits` | `[B, P, {self._relation_width}]` float32 | "
            "Relation scores, column 0 is the adaptive threshold |\n\n"
            "### ONNX Runtime, without this package\n\n"
            "`mention_mask` must be supplied by the caller. Building it from "
            "raw text means decoding the NER output first, which is what the "
            "`extract` command does for you.\n\n"
            "```python\n"
            "import numpy as np\n"
            "import onnxruntime\n"
            "from transformers import AutoTokenizer\n"
            "\n"
            'bundle = "artifacts"\n'
            "session = onnxruntime.InferenceSession(\n"
            '    f"{bundle}/model_int8.onnx", providers=["CPUExecutionProvider"]\n'
            ")\n"
            "tokenizer = AutoTokenizer.from_pretrained(bundle)\n"
            "\n"
            'words = ["Skai", "TV", "is", "a", "Greek", "television", "network"]\n'
            "encoding = tokenizer(\n"
            "    [words], is_split_into_words=True, truncation=True, max_length=512\n"
            ")\n"
            "\n"
            "mention_mask = np.zeros((1, 2, len(encoding['input_ids'][0])), "
            'dtype="float32")\n'
            "mention_mask[0, 0, 1:3] = 0.5\n"
            "mention_mask[0, 1, 5:6] = 1.0\n"
            "\n"
            "ner_logits, relation_logits = session.run(\n"
            '    ["ner_logits", "relation_logits"],\n'
            "    {\n"
            '        "input_ids": np.asarray(encoding["input_ids"], dtype="int64"),\n'
            '        "attention_mask": np.asarray(\n'
            '            encoding["attention_mask"], dtype="int64"\n'
            "        ),\n"
            '        "mention_mask": mention_mask,\n'
            '        "pair_index": np.asarray([[[0, 1], [1, 0]]], dtype="int64"),\n'
            "    },\n"
            ")\n"
            "\n"
            "predicted = relation_logits[0] > relation_logits[0][:, :1]\n"
            "predicted[:, 0] = False\n"
            "```\n\n"
            "### Python, through this package\n\n"
            "```python\n"
            "from nano_re.inference import RelationExtractor\n"
            "\n"
            'extractor = RelationExtractor.from_bundle("artifacts")\n'
            'result = extractor.extract("Skai TV is a Greek network.")\n'
            "\n"
            "for entity in result.entities:\n"
            "    print(entity.index, entity.entity_type, entity.name)\n"
            "for relation in result.relations:\n"
            "    head = result.entities[relation.head].name\n"
            "    tail = result.entities[relation.tail].name\n"
            '    print(f"{head} -[{relation.label}]-> {tail} "\n'
            '          f"({relation.confidence:.2f})")\n'
            "```\n\n"
            "`label_schema.json` maps head indices back to BIO tags and Wikidata "
            "property identifiers."
        )

    def _training_details(self, training: TrainingReport) -> str:
        """Render the training configuration section.

        Args:
            training: Report produced by the trainer.

        Returns:
            Markdown text.
        """
        hyperparameters = training.hyperparameters
        rows = [
            ("Epochs", hyperparameters.get("epochs")),
            ("Encoder learning rate", hyperparameters.get("learning_rate")),
            ("Head learning rate", hyperparameters.get("head_learning_rate")),
            ("Weight decay", hyperparameters.get("weight_decay")),
            ("Warmup ratio", hyperparameters.get("warmup_ratio")),
            ("Gradient clipping", hyperparameters.get("max_grad_norm")),
            ("NER loss weight (alpha)", hyperparameters.get("ner_loss_weight")),
            (
                "Relation loss weight (beta)",
                hyperparameters.get("relation_loss_weight"),
            ),
            ("Seed", hyperparameters.get("seed")),
            ("Device", training.device),
            ("Wall clock", f"{training.total_seconds / 60:.1f} min"),
        ]
        table = "\n".join(f"| {name} | {value} |" for name, value in rows)
        return (
            "## Training\n\n"
            f"Trained on `{'`, `'.join(self._corpora)}` with a joint objective "
            "`L = alpha * L_NER + beta * L_RE`. A corpus that annotates entities "
            "but not relations trains the token head only.\n\n"
            "| Setting | Value |\n| --- | --- |\n" + table
        )

    def _limitations(self, training: TrainingReport) -> str:
        """Render the limitations section.

        Args:
            training: Report produced by the trainer.

        Returns:
            Markdown text.
        """
        return (
            "## Limitations\n\n"
            "- Documents are truncated to 512 sub-word tokens. Entities beyond "
            "that window are unreachable, which caps relation recall.\n"
            "- The relation head consumes entity clusters supplied through "
            "`mention_mask`. Training used the corpus's gold clusters; the "
            "`extract` command substitutes a surface-form clustering heuristic, "
            "so pronouns and paraphrases do not join their antecedent's "
            "cluster.\n"
            "- The corpora are encyclopaedic prose. Business documents such as "
            "invoices and contracts are a different domain, which is why "
            "structured identifiers are matched by rule instead.\n"
            "- The relation corpus is generated automatically, so its labels "
            "are incomplete and measured precision understates the true "
            "figure.\n"
            f"- The relation decision rule is tied to the "
            f"`{training.relation_objective}` objective; changing the objective "
            "changes how the logits must be read."
        )

    def _citation(self) -> str:
        """Render the dataset citation section.

        Returns:
            Markdown text.
        """
        return (
            "## Citation\n\n"
            "```bibtex\n"
            "@inproceedings{huguet-cabot-etal-2023-redfm,\n"
            '    title = "{RED}$^{\\rm FM}$: a Filtered and Multilingual '
            'Relation Extraction Dataset",\n'
            "    author = {Huguet Cabot, Pere-Llu{\\'i}s and Tedeschi, Simone "
            "and Ngonga Ngomo, Axel-Cyrille and Navigli, Roberto},\n"
            '    booktitle = "Proceedings of the 61st Annual Meeting of the '
            'Association for Computational Linguistics",\n'
            "    year = 2023,\n"
            "}\n"
            "@inproceedings{tedeschi-navigli-2022-multinerd,\n"
            '    title = "{M}ulti{NERD}: A Multilingual, Multi-Genre and '
            'Fine-Grained Dataset for Named Entity Recognition",\n'
            "    author = {Tedeschi, Simone and Navigli, Roberto},\n"
            '    booktitle = "Findings of NAACL 2022",\n'
            "    year = 2022,\n"
            "}\n"
            "```"
        )
