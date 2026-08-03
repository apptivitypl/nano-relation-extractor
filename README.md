# Nano Relation Extractor

Multilingual entity and relation extraction from plain text, on CPU.

Built as the input stage for a knowledge graph: documents in, typed nodes and
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

Runs locally. No API key, no account, nothing uploaded.

## Status

The pipeline is complete and tested. **No trained model ships yet**, and no
benchmark numbers exist. Train one with `nano-re all`.

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-org/nano-relation-extractor
cd nano-relation-extractor
uv sync
```

## Extract

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
rule and verified by checksum, not predicted by the model. No NLP corpus
contains a NIP; a checksum gives an exact answer.

```python
from nano_re.patterns import PatternExtractor

PatternExtractor().extract("NIP 5252248481")   # valid, reported
PatternExtractor().extract("NIP 5252248482")   # bad checksum, returns []
```

Covered: NIP, REGON, KRS, PESEL, IBAN, Polish bank account, invoice number,
document number, amount, date, email, phone. Rules are data in
`patterns/library.py`; add your own by appending to a list.

## Train

```bash
uv run nano-re all
```

Stages also run alone, reading and writing the artifact directory:

```bash
uv run nano-re prepare     # download corpora, derive the label schema
uv run nano-re train       # train both heads
uv run nano-re export      # ONNX export, INT8 quantisation
uv run nano-re benchmark   # CPU latency and the accuracy cost of quantisation
uv run nano-re package     # model card and manifest
```

Wiring check, about one minute:

```bash
uv run nano-re all --limit 40 --epochs 1 --eval-split dev
```

### Hardware

The device is detected and configured automatically: CUDA, then Apple Silicon,
then CPU. Each gets settings that were measured, not assumed.

| | CUDA | Apple Silicon | CPU |
| --- | --- | --- | --- |
| Precision | bfloat16 where supported, else float16 | float32 | float32 |
| Batch size | 16 | 8 | 4 |
| Loader workers | 4 | 0 | 0 |
| Pinned memory | yes | no | no |
| TF32 matmul | enabled | n/a | n/a |

Mixed precision is not a universal win. On an M4 Pro a training step took 341 ms
in float32 and 369 ms under float16 autocast: the cast operations cost more than
the cheaper arithmetic saves on unified memory. It is therefore enabled on CUDA
only, where bfloat16 also removes the need for gradient scaling.

Override anything with `NANO_RE_TRAIN_BATCH_SIZE`, `NANO_RE_EVAL_BATCH_SIZE` and
`NANO_RE_NUM_WORKERS`. Zero, or negative for workers, keeps the measured default.

### Scale

The full SREDFM training split is 33 GB across eight languages, 5.6 million
documents. Documents are not held in memory: each file is scanned once to record
where every record begins, and records are read from disk on demand. Measured,
indexing 60,000 documents costs 20 MB resident, so the full split needs roughly
2 GB of index rather than the 57 GB materialising would take.

Time bounds a run, not memory. Measured on an M4 Pro at batch 8, sequence 512,
with context pooling on: 75 ms per document.

```bash
NANO_RE_LANGUAGES=pl,en uv run nano-re all --limit 5000 --epochs 1   # ~20 min
uv run nano-re all --limit 30000 --epochs 3                          # ~4 h
```

KPWr caps at 13,959 sentences and Re-DocRED at roughly 3,000 documents, so a
limit above those only draws more from SREDFM.

`--limit` applies per corpus. `nano-re all` is a batch job; the process exits
when the last stage finishes.

### Output bundle

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

`notebooks/train_quantize_package.ipynb` runs the same stages cell by cell.

## Data

Every corpus permits commercial use. A model trained on non-commercial data is a
model nobody can deploy.

| Corpus | Licence | Languages | Supervises |
| --- | --- | --- | --- |
| [SREDFM](https://huggingface.co/datasets/Babelscape/SREDFM) | CC BY-SA 4.0 | 18 | entities, relations |
| [Re-DocRED](https://huggingface.co/datasets/tonytan48/Re-DocRED) | MIT | English | entities, relations |
| [KPWr](https://huggingface.co/datasets/clarin-pl/kpwr-ner) | CC BY 3.0 | Polish | entities |
| [REDFM](https://huggingface.co/datasets/Babelscape/REDFM) | CC BY-SA 4.0 | 7 | evaluation |

Corpora are interleaved by weight, not concatenated. A corpus without relation
annotations is masked out of the relation loss so it cannot teach the relation
head that every pair is unrelated.

Rejected: MultiNERD and WikiNEuRal are CC BY-NC-SA. WikiANN declares no licence.
MultiCoNER v2 has no Polish.

Re-DocRED is the corrected release of DocRED, carrying 34.6 gold relations per
document against the original's 12.3.

## Architecture

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
layers, hidden size 384, 8192 token window, MIT. Swap it with
`NANO_RE_BACKBONE`.

**Entity types.** `PER`, `ORG`, `LOC`, `DATE`, `TIME`, `NUMBER`, `MEDIA`, `EVE`,
`MISC`, folded from each corpus's own inventory.

**Relations.** Wikidata properties observed in the corpora. The inventory is
counted during a first pass and frozen into `label_schema.json`, so head width
and decoded names cannot drift apart.

**Relation head.** Adaptive thresholding with localized context pooling, after
Zhou et al. (AAAI 2021). Class zero is a threshold learned per pair, so there is
no global probability cutoff to tune; gold relations are about three percent of
candidate pairs and a plain binary objective collapses to predicting nothing.
The context vector is built from the tokens both entities attend to, which is
what tells the head which part of the document connects them.
`NANO_RE_RELATION_LOSS=bce` selects the binary alternative.

**Export.** Verification is part of the export. The graph is compared against
PyTorch on three differently shaped batches, and export fails if relative
deviation exceeds tolerance or if the two would ever choose different classes.

**Quantisation.** Dynamic INT8 over `MatMul` and `Gather`. It reliably cuts file
size by about four. It does not reliably cut latency: faster on x86 with VNNI,
level with float32 on Apple Silicon. The benchmark measures both graphs on your
hardware and the model card reports what actually happened.

## Configuration

Frozen dataclasses in `config.py`, each with an environment override.

| Variable | Default | Effect |
| --- | --- | --- |
| `NANO_RE_LANGUAGES` | `pl,en,de,fr,es,it,nl,pt` | Languages read from the corpora |
| `NANO_RE_BACKBONE` | `jhu-clsp/mmBERT-small` | Any Hugging Face encoder |
| `NANO_RE_OUTPUT_DIR` | `artifacts` | Bundle destination |
| `NANO_RE_EPOCHS` | `3` | Training epochs |
| `NANO_RE_MAX_SEQUENCE_LENGTH` | `512` | Encoder window in sub-words |
| `NANO_RE_TRAIN_BATCH_SIZE` | auto | Zero keeps the per-device default |
| `NANO_RE_NUM_WORKERS` | auto | Negative keeps the per-device default |
| `NANO_RE_RELATION_WEIGHT` | `4.0` | Sampling weight of SREDFM |
| `NANO_RE_ENTITY_WEIGHT` | `1.0` | Sampling weight of KPWr |
| `NANO_RE_ENGLISH_RELATION_WEIGHT` | `1.0` | Sampling weight of Re-DocRED |
| `NANO_RE_LOCALIZED_CONTEXT` | `true` | Pair context from encoder attention |
| `NANO_RE_MAX_RELATIONS` | unset | Cap the inventory to the most frequent |
| `NANO_RE_TRIM_VOCABULARY` | `false` | Compact the embedding table |
| `NANO_RE_RELATION_LOSS` | `adaptive_threshold` | Or `bce` |

Context pooling materialises the encoder's attention maps, which grow with the
square of sequence length. Above roughly 1024 tokens, turn it off or accept the
memory cost.

### Vocabulary trimming

The embedding table is 98.3M of mmBERT-small's 140.5M parameters. Trimming it to
the tokens your languages use cuts the model by roughly four.

Off by default: a released multilingual model should work in the languages it
advertises, and trimming degrades every language left out. Turn it on for a
fixed language set:

```bash
NANO_RE_TRIM_VOCABULARY=true NANO_RE_LANGUAGES=pl,en uv run nano-re all
```

If you do, sample your own documents into the token count as well, or rare names
and domain abbreviations will fall back to the unknown token.

## Licensing

Code is dual licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT),
at your option.

Weights are separate. They inherit obligations from their training data, which
by default includes SREDFM and REDFM under CC BY-SA 4.0. For permissive weights,
train on the permissive subset:

```bash
NANO_RE_RELATION_WEIGHT=0 uv run nano-re all
```

That excludes SREDFM and REDFM entirely, including from the label schema, and
trains on Re-DocRED (MIT) and KPWr (CC BY 3.0). It costs sixteen languages of
relation supervision. The model card records which corpora went in.

## Tests

```bash
uv run pytest
```

Covers the places where a mistake produces no error: identifier checksums, BIO
decoding at span boundaries, mention clustering, character offset alignment,
corpus interleaving ratios, task masking, context pooling, the vocabulary remap,
and random access into indexed corpora.

## Limitations

- **Relation quality is bounded by the data.** SREDFM is generated
  automatically, so labels are noisy and incomplete. Re-DocRED is human
  corrected but English only. No gold relation evaluation exists for Polish, so
  Polish relation quality cannot be measured.
- **Relations are encyclopaedic.** Business specific relations such as "party to
  this contract" appear in no public corpus.
- **Coreference is heuristic.** Entity clusters are built by matching normalised
  surface forms with a whole word prefix rule. A pronoun starts its own cluster.
  Relation quality depends on this directly.
- **No entity linking.** The model returns typed mentions, not Wikidata
  identifiers.
- **No cross-document merging.** Clustering works within a single input.
- **Rules target Polish and EU identifiers.** Other schemes need their own
  entries in `patterns/library.py`.
