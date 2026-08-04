# Nano Relation Extractor

Multilingual entity and relation extraction from plain text, on CPU.

Built as the stage in front of a knowledge graph: documents in, typed nodes and
typed edges out, without running a large language model over every document.

```
"Skai TV is a Greek television network based in Piraeus. It is part of Skai Group."

Entities (4)
  [0] ORG   Skai TV      x2
  [1] LOC   Greek
  [2] LOC   Piraeus
  [3] ORG   Skai Group

Relations (2)
  Skai TV --[country]--> Greek                  (0.91)
  Skai TV --[headquarters location]--> Piraeus  (0.87)
```

Everything runs locally. No API key, no account, nothing uploaded.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apptivitypl/nano-relation-extractor/blob/main/notebooks/train_quantize_package.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/apptivitypl/nano-relation-extractor/blob/main/notebooks/train_quantize_package.ipynb)

> **Status.** The pipeline is complete and tested. No trained model ships yet and
> no benchmark numbers exist. Train one with `nano-re all`.

## Contents

- [Quick start](#quick-start)
- [Extracting](#extracting)
- [Training](#training)
- [Training data](#training-data)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Licensing](#licensing)
- [Limitations](#limitations)

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/apptivitypl/nano-relation-extractor
cd nano-relation-extractor
uv sync
```

Check the wiring in about a minute:

```bash
uv run nano-re all --limit 40 --epochs 1
```

Then train something real:

```bash
uv run nano-re all --limit 30000 --epochs 3
```

Nothing else needs configuring. The device, batch size, gradient accumulation
and relation inventory are all derived from what the machine and the corpora
turn out to be.

## Extracting

```bash
uv run nano-re extract
```

Interactive: paste text, blank line to submit, Ctrl-D to quit.

```bash
uv run nano-re extract --text "Skai TV is a Greek network based in Piraeus."
uv run nano-re extract --file article.txt --json
cat corpus.txt | uv run nano-re extract --json --top-k 20
```

| Flag | Effect |
| --- | --- |
| `--backend` | `onnx-int8` (default), `onnx-fp32`, `pytorch` |
| `--json` | Machine readable output |
| `--top-k` | Report at most N relations |
| `--min-confidence` | Drop relations below this score |

From Python:

```python
from nano_re.inference import RelationExtractor

extractor = RelationExtractor.from_bundle("artifacts")
result = extractor.extract(open("contract.txt").read())

for entity in result.entities:
    print(entity.index, entity.entity_type, entity.name)

for relation in result.relations:
    head = result.entities[relation.head].name
    tail = result.entities[relation.tail].name
    print(f"{head} -[{relation.label}]-> {tail}  {relation.confidence:.2f}")
```

Input length is unbounded. Longer text is split into overlapping windows and the
results merged, so an entity crossing a boundary stays one entity.

### Structured identifiers

Tax numbers, bank accounts, invoice numbers, amounts and dates are matched by
rule and verified by checksum rather than predicted. No NLP corpus contains a
NIP, and a checksum gives an exact answer where a model would guess.

```python
from nano_re.patterns import PatternExtractor

PatternExtractor().extract("NIP 5252248481")   # valid, reported
PatternExtractor().extract("NIP 5252248482")   # bad checksum, returns []
```

Covered: NIP, REGON, KRS, PESEL, IBAN, Polish bank account, invoice number,
document number, amount, date, email, phone. Rules are plain data in
`patterns/library.py`; add a scheme by appending to a list.

## Training

```bash
uv run nano-re all
```

Stages also run alone, each reading and writing the artifact directory:

```bash
uv run nano-re prepare     # download corpora, derive the label schema
uv run nano-re train       # train both heads
uv run nano-re export      # ONNX export, INT8 quantisation
uv run nano-re benchmark   # CPU latency, and the accuracy cost of quantisation
uv run nano-re package     # model card and manifest
```

`--limit` caps documents per corpus and is the one dial worth turning. Only that
many are downloaded, so it governs disk, network and time together. A capped run
fetches a truncated copy of each file over HTTP: `--limit 5000` on Polish
transfers about 30 MB rather than the 3.3 GB the split weighs, and starts in
seconds.

| `--limit` | Documents per epoch | Download | On an M4 Pro, 3 epochs |
| --- | --- | --- | --- |
| 5000 | ~13k | ~150 MB | ~45 min |
| 30000 | ~47k | ~900 MB | ~5 h |
| unset | 5.6M | 33 GB | days |

`nano-re all` is an ordinary batch job. It runs the stages in order and exits
when the last finishes. Progress is reported per batch, with a rate and an
estimate, so a long run is never silent.

### Hardware

The device is detected and configured automatically: CUDA, then Apple Silicon,
then CPU.

| | CUDA | Apple Silicon | CPU |
| --- | --- | --- | --- |
| Precision | bfloat16, or float16 with scaling on older cards | float32 | float32 |
| Batch | from card memory, then verified | 8 | 4 |
| Loader workers | half the cores, capped at 8 | 0 | 0 |
| Pinned memory | yes | no | no |
| TF32 matmul | enabled | n/a | n/a |

Mixed precision is not a universal win. On an M4 Pro a training step took 341 ms
in float32 and 369 ms under float16 autocast, because the casts cost more than
the cheaper arithmetic saves on unified memory. It is enabled on CUDA only.

Before training starts, one real step is attempted at the chosen batch size and
the batch is halved until it fits. A static table cannot predict this: memory
depends on sequence length, entity count and whether context pooling is on. The
probe costs seconds and turns an out of memory failure an hour into a run into a
decision made before it begins.

Gradient accumulation then makes up whatever the batch could not, so a card
holding eight documents takes the same optimisation step as one holding
thirty-two, and the learning rate means the same thing on every machine.

### Free GPU

`notebooks/train_quantize_package.ipynb` runs the whole pipeline a stage per
cell, and runs anywhere. Locally it uses the checkout it was started from; on
Kaggle or Colab it clones the repository and installs what the base image lacks.
The badges above open it directly.

On Kaggle, set the accelerator to a T4 or P100 and turn internet on before
running. Two of its limits shape the run and both are handled: sessions stop at
twelve hours, so choose a limit that leaves headroom, and the working directory
holds about 20 GB, so the caches point at the larger scratch volume and only the
records the run needs are downloaded.

### Memory

Documents are never held in memory. Each corpus file is scanned once to record
where every record begins, and records are read from disk on demand. Indexing
60,000 documents was measured at 20 MB resident, so the full 5.6 million document
split needs roughly 2 GB of index rather than the 57 GB that materialising parsed
documents would take.

Time bounds a run, not memory.

### Output

```
artifacts/
  model_int8.onnx        quantised graph, the deployment artifact
  model.onnx             float32 graph, kept for comparison
  model.safetensors      PyTorch weights
  config.json            architecture description
  tokenizer.json         tokenizer
  label_schema.json      entity tags and relation inventory
  MODEL_CARD.md          generated from measurements
  MANIFEST.json          file inventory
  training_report.json   per-epoch losses and scores
  export_report.json     export verification and quantisation results
  benchmark.json         CPU latency and accuracy comparison
```

## Training data

Every corpus permits commercial use. A model trained on non-commercial data is a
model nobody can deploy, which ruled out several otherwise attractive datasets.

| Corpus | Licence | Languages | Supervises |
| --- | --- | --- | --- |
| [SREDFM](https://huggingface.co/datasets/Babelscape/SREDFM) | CC BY-SA 4.0 | 18 | entities, relations |
| [Re-DocRED](https://huggingface.co/datasets/tonytan48/Re-DocRED) | MIT | English | entities, relations |
| [KPWr](https://huggingface.co/datasets/clarin-pl/kpwr-ner) | CC BY 3.0 | Polish | entities |
| [REDFM](https://huggingface.co/datasets/Babelscape/REDFM) | CC BY-SA 4.0 | 7 | evaluation |

Corpora are interleaved by weight rather than concatenated, so both heads keep
receiving signal throughout an epoch. A corpus that annotates entities but not
relations is masked out of the relation loss, so it cannot teach the relation
head that every pair is unrelated.

Re-DocRED is the corrected release of DocRED and carries 34.6 gold relations per
document against the original's 12.3, so training against the original punishes a
model for predictions that are in fact right.

Rejected: MultiNERD and WikiNEuRal are CC BY-NC-SA. WikiANN declares no licence.
MultiCoNER v2 has no Polish.

## How it works

```
text
 |- pattern rules --------------------------> identifiers (checksum verified)
 `- windowing --> encoder --> token head ---> entity spans
                     |                             |
                     |                        clustering
                     |                             |
                     `-------------> relation head --> typed relations
```

Extraction runs the model twice: the first pass tags entities, the second scores
relations using pooling weights built from those tags. The relation head's input
depends on the token head's output, so this is not avoidable.

**Encoder.** [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small): 22
layers, hidden size 384, an 8192 token window, MIT licensed. Substitute any
Hugging Face encoder with `NANO_RE_BACKBONE`.

**Entity types.** `PER`, `ORG`, `LOC`, `DATE`, `TIME`, `NUMBER`, `MEDIA`, `EVE`,
`MISC`, folded from each corpus's own inventory.

**Relations.** Wikidata properties observed in the corpora. The inventory is
counted during a first pass and frozen into `label_schema.json`, so head width
and decoded names cannot drift apart. Its tail is pruned by coverage rather than
by a chosen threshold: eight languages turn up over six hundred predicates, more
than half with fewer than ten examples, which cannot be learned and only dilute
the averaged score.

**Relation head.** Adaptive thresholding with localized context pooling, after
Zhou et al. (AAAI 2021). Class zero is a threshold learned per pair, so there is
no global probability cutoff to tune; gold relations are about three percent of
candidate pairs and a plain binary objective collapses to predicting nothing. The
context vector is built from the tokens both entities attend to, which is what
tells the head which part of the document connects them.

**Coreference.** The relation head consumes entity clusters. Training corpora
supply gold clusters; at inference they come from matching normalised surface
forms, with a whole word prefix rule so "Skai" joins "Skai TV".

**Export.** Verification is part of the export. The graph is compared against
PyTorch on three differently shaped batches, and export fails if relative
deviation exceeds tolerance or if the two implementations would ever choose
different classes.

**Quantisation.** Dynamic INT8 over `MatMul` and `Gather`. It reliably cuts file
size by about four. It does not reliably cut latency: faster on x86 with VNNI,
level with float32 on Apple Silicon. The benchmark measures both graphs on your
hardware and the model card reports what actually happened.

## Configuration

Frozen dataclasses in `config.py`, each with an environment override. Defaults
are what the commands above run.

| Variable | Default | Effect |
| --- | --- | --- |
| `NANO_RE_LANGUAGES` | `pl,en,de,fr,es,it,nl,pt` | Languages read from the corpora |
| `NANO_RE_BACKBONE` | `jhu-clsp/mmBERT-small` | Any Hugging Face encoder |
| `NANO_RE_OUTPUT_DIR` | `artifacts` | Bundle destination |
| `NANO_RE_EPOCHS` | `3` | Training epochs |
| `NANO_RE_MAX_SEQUENCE_LENGTH` | `512` | Encoder window in sub-words |
| `NANO_RE_RELATION_WEIGHT` | `4.0` | Sampling weight of SREDFM |
| `NANO_RE_ENTITY_WEIGHT` | `1.0` | Sampling weight of KPWr |
| `NANO_RE_ENGLISH_RELATION_WEIGHT` | `1.0` | Sampling weight of Re-DocRED |
| `NANO_RE_RELATION_COVERAGE` | `0.999` | Share of relation instances the kept classes cover |
| `NANO_RE_LOCALIZED_CONTEXT` | `true` | Pair context from encoder attention |
| `NANO_RE_RELATION_LOSS` | `adaptive_threshold` | Or `bce` |
| `NANO_RE_TRIM_VOCABULARY` | `false` | Compact the embedding table |
| `NANO_RE_TRAIN_BATCH_SIZE` | auto | Zero keeps the per-device value |
| `NANO_RE_NUM_WORKERS` | auto | Negative keeps the per-device value |

Context pooling materialises the encoder's attention maps, which grow with the
square of sequence length. Above roughly 1024 tokens, turn it off or accept the
memory cost.

### Vocabulary trimming

The embedding table is 98.3M of mmBERT-small's 140.5M parameters. Trimming it to
the tokens your languages use cuts the model by roughly four.

It is off by default, because a released multilingual model should work in the
languages it advertises and trimming degrades every language left out. Turn it
on for a fixed language set:

```bash
NANO_RE_TRIM_VOCABULARY=true NANO_RE_LANGUAGES=pl,en uv run nano-re all
```

If you do, sample your own documents into the token count as well, or rare names
and domain abbreviations will fall back to the unknown token.

## Licensing

Code is dual licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT),
at your option.

Weights are a separate question. They inherit obligations from their training
data, which by default includes SREDFM and REDFM under CC BY-SA 4.0. For weights
under permissive terms only, train on the permissive subset:

```bash
NANO_RE_RELATION_WEIGHT=0 uv run nano-re all
```

That excludes SREDFM and REDFM entirely, including from the label schema, and
trains on Re-DocRED (MIT) and KPWr (CC BY 3.0) alone. It costs sixteen languages
of relation supervision. The model card records which corpora went in.

## Tests

```bash
uv run pytest
```

The suite covers the places where a mistake produces no error rather than a
crash: identifier checksums, BIO decoding at span boundaries, mention clustering,
character offset alignment, corpus interleaving ratios, task masking in the loss,
context pooling, the vocabulary remap, batch probing, and random access into
indexed corpora.

## Limitations

- **Relation quality is bounded by the data.** SREDFM is generated
  automatically, so its labels are noisy and incomplete. Re-DocRED is human
  corrected but English only. No gold relation evaluation exists for Polish, in
  this project or anywhere else, so Polish relation quality cannot be measured.
- **Relations are encyclopaedic.** Business specific relations such as "party to
  this contract" appear in no public corpus.
- **Coreference is heuristic.** A pronoun starts its own cluster instead of
  joining its antecedent, and relation quality depends on clustering directly.
- **No entity linking.** The model returns typed mentions, not Wikidata
  identifiers.
- **No cross-document merging.** Clustering works within a single input; a
  corpus level entity registry belongs to whatever consumes this output.
- **Rules target Polish and EU identifiers.** Other schemes need their own
  entries in `patterns/library.py`.
