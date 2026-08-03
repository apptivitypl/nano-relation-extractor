# Nano Relation Extractor

A multi-task pipeline that trains one lightweight model to do named entity
recognition and document-level relation extraction in a single forward pass,
exports it to ONNX, quantises it to INT8 for CPU inference, and assembles
everything into a self-contained bundle on your disk with a generated model
card.

The model shares a `nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large`
encoder between a BIO token classifier and a pairwise relation classifier, and
is trained on [DocRED](https://huggingface.co/datasets/thunlp/docred).

**Everything stays local, and there is nothing to configure to get started.**
Nothing is uploaded anywhere and no credential is required: the Hugging Face Hub
is used only to download the public DocRED corpus and the public pretrained
encoder.

## Environment

```bash
uv init --package --name nano-re --python 3.11
```

```bash
uv venv
```

```bash
uv add torch transformers datasets accelerate onnx onnxruntime onnxscript optimum huggingface_hub jupyterlab scikit-learn evaluate seqeval
```

Two dependencies are not in the minimal list. `onnxscript` is required by the
`torch.onnx` dynamo exporter; without it the export silently falls back to the
deprecated TorchScript path. `seqeval` backs `evaluate.load("seqeval")` for
span-level NER scoring.

```bash
uv run jupyter lab
```

## Running the pipeline

Every stage reads its inputs from the artifact directory and writes its outputs
back, so stages can be run individually or resumed.

```bash
uv run nano-re all
```

```bash
uv run nano-re prepare
```

```bash
uv run nano-re train --epochs 3
```

```bash
uv run nano-re export
```

```bash
uv run nano-re benchmark
```

```bash
uv run nano-re package
```

`package` renders `MODEL_CARD.md` from the measured reports, inventories the
bundle, checks that nothing expected is missing and writes `MANIFEST.json`. Pass
`--drop-fp32-graph` to delete the float32 ONNX graph, which is a quantisation
intermediate costing about four times the INT8 graph on disk; keep it if you
want to re-run the benchmark comparison later.

The finished bundle in `artifacts/`:

```
artifacts/
  model.safetensors        Trained PyTorch weights
  config.json              Architecture description for rebuilding the model
  model_int8.onnx          INT8 graph for CPU inference
  model.onnx               Float32 graph, kept for benchmark comparison
  tokenizer.json           Tokenizer
  tokenizer_config.json
  label_schema.json        BIO tag and relation vocabularies
  training_report.json     Per-epoch losses and evaluation scores
  export_report.json       Export verification and quantisation results
  benchmark.json           CPU latency and accuracy comparison
  MODEL_CARD.md            Generated documentation
  MANIFEST.json            File inventory with sizes
```

A fast wiring check that downloads about one megabyte and finishes in a minute:

```bash
uv run nano-re all --limit 12 --epochs 1 --train-split dev --eval-split dev
```

`notebooks/train_quantize_package.ipynb` drives the same stages cell by cell.

## Using the trained model

Once a bundle exists, `extract` runs it over your own text. With no input
argument it starts an interactive session: paste text, press Enter on an empty
line, get the extraction. Ctrl-D quits.

```bash
uv run nano-re extract
```

```bash
uv run nano-re extract --text "Skai TV is a Greek television network based in Piraeus."
```

```bash
uv run nano-re extract --file article.txt --json
```

Text also arrives over a pipe, which makes the command scriptable:

```bash
cat article.txt | uv run nano-re extract --json --top-k 20
```

Output:

```
Entities (3)
  [0] ORG   Skai TV  x2
  [1] LOC   Greek
  [2] LOC   Piraeus

Relations (2)
  Skai TV --[country]--> Greek   (0.91)
  Skai TV --[headquarters location]--> Piraeus   (0.87)
```

| Flag | Effect |
| --- | --- |
| `--backend` | `onnx-int8` (default), `onnx-fp32` or `pytorch` |
| `--json` | Machine readable output instead of the table |
| `--top-k` | Report at most N relations, best first |
| `--min-confidence` | Drop relations below this score |

### How extraction closes the gap the model leaves open

The relation head is trained on DocRED's gold coreference clusters, delivered
through `mention_mask`. Raw text has no such clusters, so `extract` has to build
them, and it does so in four steps:

1. Run the model to get BIO logits.
2. Decode BIO tags into typed mention spans.
3. **Cluster mentions into entities** by normalised surface form and type, with
   a whole-word prefix rule so "Skai" joins "Skai TV".
4. Build `mention_mask` and every ordered pair from those clusters, then run the
   model a second time for the relation logits.

Step 3 is a heuristic standing in for a coreference model that this pipeline
never trains. It suits encyclopaedic prose, where entities recur verbatim, and
it is the weakest link in end-to-end extraction: a pronoun referring back to an
entity starts its own cluster instead of joining it. Step 4 is why extraction
costs two forward passes rather than one — the input to the relation head
depends on the output of the NER head.

## Design

```
src/nano_re/
  config.py     Frozen dataclasses; the only place that reads the environment
  schema.py     BIO and relation vocabularies shared by every stage
  data/         source -> parser -> encoder -> collator -> DataModule
  models/       backbone + heads, composed by NanoREModel, built by a factory
  training/     device policy, losses, metrics, evaluator, trainer
  export/       ONNX exporter, INT8 quantiser, runtime adapter, benchmark
  artifacts/    model card builder, bundle assembler
  inference/    text splitter, BIO decoder, clusterer, backends, console
  pipeline.py   Composition root sequencing the stages
  cli.py        Thin argparse wrapper over the pipeline and the extractor
```

Each stage depends on the abstraction above it, not on its neighbours. Swapping
the corpus means writing a new `DocumentSource` and parser; swapping the relation
decision rule means writing a new `RelationObjective`; scoring the ONNX graph
reuses the PyTorch evaluator through `OnnxModelAdapter` rather than duplicating
an evaluation loop.

### Data

`thunlp/docred` is a loading-script dataset with no Parquet conversion, and
`datasets` 3.0 removed script execution, so `load_dataset` cannot read it on any
current release. `DocREDHubSource` downloads the gzipped JSON archives directly
and rebuilds `datasets.Dataset` objects, which keeps the familiar API and avoids
`trust_remote_code` entirely.

DocRED stores mention offsets relative to their sentence; the parser converts
them to document-global word indices once, so no downstream component has to.

### Model

```
input_ids [B,S], attention_mask [B,S], mention_mask [B,E,S], pair_index [B,P,2]
  -> shared encoder                                  -> hidden [B,S,384]
  -> token classification head                       -> ner_logits [B,S,13]
  -> entity pooling (mention_mask @ hidden)          -> entities [B,E,384]
  -> relation head over [h; t; h*t; |h-t|]           -> re_logits [B,P,97]
```

Entity pooling is a batched matrix product and pair selection is a `gather` with
an expanded index. Neither introduces data-dependent control flow, which is what
lets all four axes stay dynamic in the exported graph.

### Relation objective

DocRED averages 12 gold triples against roughly 380 candidate pairs per
document, a three percent positive rate. The default objective is adaptive
thresholding: class 0 is a threshold the network learns per pair, positives are
ranked above it and negatives below it. This removes the global probability
threshold that a plain binary objective has to have tuned. `--relation-loss bce`
selects the binary alternative, whose threshold `ThresholdSearch` can tune on the
evaluation split.

### Export

The exporter treats verification as part of the export. It compares ONNX Runtime
logits against PyTorch on the traced batch, then repeats the comparison with
different batch, sequence, entity and pair counts. An export that silently froze
a shape would pass the first check and fail the second, so both are hard gates.

Quantisation preprocessing tries symbolic shape inference first and falls back to
ONNX shape inference alone: transformer encoders build position identifiers with
a `Range` node whose limit is symbolic under a dynamic sequence axis, which the
symbolic pass cannot evaluate.

`Gather` is quantised alongside `MatMul` because the 250k multilingual
vocabulary holds roughly ninety percent of the parameters. Leaving it in float32
caps the size reduction at a few percent.

### A note on INT8 latency

Dynamic INT8 quantisation is a dependable size win and a conditional speed win.
It reduces the graph from about 432 MB to about 109 MB, a factor of four, on any
host. Latency depends on whether the host has integer dot-product kernels: on
x86 with VNNI it is a clear improvement, while on Apple Silicon the measured
median is roughly level with float32 because dequantisation overhead offsets the
cheaper arithmetic. The benchmark measures both graphs and the generated model
card reports whichever direction actually occurred, rather than assuming a
speedup.

## Configuration

All settings come from frozen dataclasses in `config.py`, each with a `from_env`
constructor, so every field has a working default and an environment override.
The defaults are what the commands above run; these are the overrides worth
knowing about.

| Variable | Default | Effect |
| --- | --- | --- |
| `NANO_RE_OUTPUT_DIR` | `artifacts` | Where the bundle is written |
| `NANO_RE_MODEL_NAME` | `nano-relation-extractor` | Name in the model card |
| `NANO_RE_TRAIN_SPLIT` | `train_annotated` | `train_distant` for the 101k distant split |
| `NANO_RE_EVAL_SPLIT` | `dev` | Evaluation split |
| `NANO_RE_EPOCHS` | `3` | Training epochs |
| `NANO_RE_TRAIN_BATCH_SIZE` | `4` | Documents per batch |
| `NANO_RE_RELATION_LOSS` | `adaptive_threshold` | Or `bce` |
| `NANO_RE_QUANTIZED_OP_TYPES` | `MatMul,Gather` | ONNX ops eligible for INT8 |
| `NANO_RE_KEEP_FP32_GRAPH` | `true` | Retain the float32 graph when packaging |
| `NANO_RE_INTRA_OP_NUM_THREADS` | `0` | ONNX Runtime CPU threads, `0` is automatic |

### Credentials

There are none. The project reads no secret, writes no secret and has no `.env`
to fill in. If you happen to have `HF_TOKEN` exported or have run
`huggingface-cli login`, `huggingface_hub` picks it up on its own and downloads
authenticated, which only lifts the anonymous rate limit. Nothing in this
codebase reads, forwards or stores it.

## Limitations

- Documents are truncated to 512 sub-word tokens, which caps relation recall.
  The recall ceiling is measured and reported rather than hidden.
- The relation head consumes gold entity spans through `mention_mask`. For raw
  text, `extract` supplies them from a surface-form clustering heuristic rather
  than a trained coreference model, so pronouns and paraphrases do not join
  their antecedent's cluster.
- `train_distant` is supported by configuration but not exercised by default.
- Evidence sentence prediction, DocRED's third subtask, is not implemented.
