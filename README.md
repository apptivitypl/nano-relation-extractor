# nano-relation-extractor

Multilingual extraction of entities and the relations between them, from plain
text, on CPU.

It is built for the step in front of a knowledge graph: you have documents, you
need typed nodes and typed edges, and running a large language model over every
document is too slow or too expensive to be the answer.

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

Everything runs locally. Nothing is uploaded, no API key is needed, and no
account is required.

## What you get

**One model, two tasks.** A shared encoder feeds a token classifier that finds
entity spans and a pairwise classifier that scores the relation between every
pair of entities. Both run in the same forward pass, so you pay for the encoder
once.

**Nine entity types and several hundred relation types.** Entities are the ones
worth putting in a graph: `PER`, `ORG`, `LOC`, `DATE`, `TIME`, `NUMBER`,
`MEDIA`, `EVE`, `MISC`. Relations come from Wikidata properties observed in the
training corpora, things like `country`, `employer`, `owner of`, `parent
organization`, `headquarters location`.

**Eight languages by default**, and up to eighteen if you configure them:
Polish, English, German, French, Spanish, Italian, Dutch, Portuguese out of the
box, with Arabic, Catalan, Greek, Hindi, Japanese, Korean, Russian, Swedish,
Vietnamese and Chinese available from the same corpora.

**Structured identifiers by rule, not by guesswork.** Tax numbers, bank
accounts, invoice numbers and amounts are matched by regular expression and
verified by checksum. A model trained on encyclopaedic text has never seen a
NIP; a checksum has an exact answer. Fabricated identifiers are rejected rather
than reported:

```python
from nano_re.patterns import PatternExtractor

extractor = PatternExtractor()
extractor.extract("NIP 5252248481")   # valid, reported
extractor.extract("NIP 5252248482")   # one digit changed, returns []
```

**Runs on CPU.** The deployment artifact is a single quantised ONNX graph. There
is no GPU requirement and no Python framework requirement at inference time
beyond ONNX Runtime and a tokenizer.

## Install

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-org/nano-relation-extractor
cd nano-relation-extractor
uv sync
```

## Use a trained model

```bash
uv run nano-re extract
```

With no arguments this opens an interactive session: paste text, press Enter on
an empty line, read the result. Ctrl-D exits.

```bash
uv run nano-re extract --text "Skai TV is a Greek network based in Piraeus."
uv run nano-re extract --file article.txt --json
cat corpus.txt | uv run nano-re extract --json --top-k 20
```

| Flag | Effect |
| --- | --- |
| `--backend` | `onnx-int8` (default), `onnx-fp32`, or `pytorch` |
| `--json` | Machine readable output |
| `--top-k` | Report at most N relations, highest confidence first |
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

Input length is not a constraint. Text longer than the encoder window is split
into overlapping windows and the results are merged, so an entity that straddles
a boundary stays one entity.

## Train your own

```bash
uv run nano-re all
```

That runs every stage in order. Each stage can also be run alone, reading its
inputs from the artifact directory and writing its outputs back:

```bash
uv run nano-re prepare     # download corpora, derive the label schema
uv run nano-re train       # train both heads
uv run nano-re export      # export to ONNX, quantise to INT8
uv run nano-re benchmark   # measure CPU latency and the cost of quantisation
uv run nano-re package     # write the model card and the manifest
```

A one minute wiring check that downloads about a megabyte:

```bash
uv run nano-re all --limit 40 --epochs 1 --eval-split dev
```

The result is a directory you can copy anywhere:

```
artifacts/
  model_int8.onnx        quantised graph, the deployment artifact
  model.onnx             float32 graph, kept for comparison
  model.safetensors      PyTorch weights
  config.json            architecture description
  tokenizer.json         tokenizer
  label_schema.json      entity tags and relation inventory
  MODEL_CARD.md          generated from measurements, not written by hand
  MANIFEST.json          file inventory
  training_report.json   per-epoch losses and scores
  export_report.json     export verification and quantisation results
  benchmark.json         CPU latency and accuracy comparison
```

`notebooks/train_quantize_package.ipynb` runs the same stages cell by cell.

## Training data

Every corpus was chosen to permit commercial use. That is a hard constraint: a
model trained on a non-commercial corpus is a model nobody can deploy, and
several otherwise attractive datasets were rejected for exactly that reason.

| Corpus | Licence | Languages | Supervises |
| --- | --- | --- | --- |
| [SREDFM](https://huggingface.co/datasets/Babelscape/SREDFM) | CC BY-SA 4.0 | 18 | entities and relations |
| [Re-DocRED](https://huggingface.co/datasets/tonytan48/Re-DocRED) | MIT | English | entities and relations |
| [KPWr](https://huggingface.co/datasets/clarin-pl/kpwr-ner) | CC BY 3.0 | Polish | entities |
| [REDFM](https://huggingface.co/datasets/Babelscape/REDFM) | CC BY-SA 4.0 | 7 | evaluation only |

Corpora are interleaved by weight rather than concatenated, so both heads keep
receiving signal throughout training. A corpus that annotates entities but not
relations is masked out of the relation loss, so it cannot teach the relation
head that every pair is unrelated.

Rejected: MultiNERD and WikiNEuRal are CC BY-NC-SA, which is non-commercial.
WikiANN has no declared licence. MultiCoNER v2 has no Polish.

Re-DocRED matters more than its size suggests. It is the corrected release of
DocRED, and the correction is large: 34.6 gold relations per document against
the original's 12.3. Training against the original punishes a model for
predictions that are in fact right.

## Licensing

**Code** is dual licensed under [Apache 2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT), at your option.

**Weights are a separate question**, and it is worth being precise about it.
Trained weights inherit obligations from the data they were trained on. With the
default configuration, that includes SREDFM and REDFM, both CC BY-SA 4.0, so
attribution and share-alike considerations apply to the weights you produce.

If you need weights under permissive terms only, train on the permissive subset:

```bash
NANO_RE_RELATION_WEIGHT=0 uv run nano-re all
```

That drops SREDFM and REDFM and trains on Re-DocRED (MIT) and KPWr (CC BY 3.0)
alone. You lose sixteen languages of relation supervision, which is a real cost.
Which trade you want is yours to make, and the model card records which corpora
went in.

## How it works

```
text
 ├── pattern rules ─────────────────────────► identifiers (checksum verified)
 └── windowing ──► encoder ──► token head ──► entity spans
                       │                          │
                       │                     clustering
                       │                          │
                       └───────────► relation head ──► typed relations
```

Extraction runs the model twice. The first pass tags entities; the second pass
scores relations using pooling weights built from those tags. That is not an
optimisation oversight, it is unavoidable: the relation head's input depends on
the token head's output.

**Encoder.** [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), 22
layers, hidden size 384, an 8192 token window, MIT licensed. Any Hugging Face
encoder can be substituted through `NANO_RE_BACKBONE`.

**Relation objective.** Adaptive thresholding, after Zhou et al. (AAAI 2021).
Class zero is a threshold the network learns per pair, so there is no global
probability cutoff to tune. This matters because gold relations are roughly
three percent of candidate pairs, and a plain binary objective collapses to
predicting nothing. Binary cross entropy is available through
`NANO_RE_RELATION_LOSS=bce` if you prefer a tunable threshold.

**Coreference.** The relation head consumes entity clusters. Training corpora
supply gold clusters; at inference they are produced by matching normalised
surface forms, with a whole word prefix rule so "Skai" joins "Skai TV". This is
a heuristic standing in for a coreference model that this project does not
train, and it is the weakest link in end to end extraction. A pronoun referring
back to an entity starts its own cluster instead of joining it.

**Export.** Verification is part of the export, not a separate step. The graph
is compared against PyTorch on three differently shaped batches, and the export
fails if the relative deviation exceeds tolerance or if the two implementations
would ever choose different classes. An export that silently freezes a shape
passes the first check and fails the others.

**Quantisation.** Dynamic INT8 over `MatMul` and `Gather`. Quantising `Gather`
is what shrinks the embedding table, which is most of the model.

## A note on INT8 latency

Quantisation reliably cuts file size by about a factor of four. It does not
reliably cut latency. On x86 with VNNI it is faster; on Apple Silicon the
measured median comes out level with float32, because dequantisation overhead
offsets the cheaper arithmetic. The benchmark measures both graphs on your
hardware and the generated model card reports whichever direction actually
occurred, rather than assuming an improvement.

## Configuration

Every setting is a frozen dataclass in `config.py` with an environment override.
The defaults are what the commands above run.

| Variable | Default | Effect |
| --- | --- | --- |
| `NANO_RE_LANGUAGES` | `pl,en,de,fr,es,it,nl,pt` | Languages read from the corpora |
| `NANO_RE_BACKBONE` | `jhu-clsp/mmBERT-small` | Any Hugging Face encoder |
| `NANO_RE_OUTPUT_DIR` | `artifacts` | Where the bundle is written |
| `NANO_RE_EPOCHS` | `3` | Training epochs |
| `NANO_RE_MAX_SEQUENCE_LENGTH` | `512` | Encoder window in sub-word tokens |
| `NANO_RE_RELATION_WEIGHT` | `4.0` | Sampling weight of SREDFM |
| `NANO_RE_ENTITY_WEIGHT` | `1.0` | Sampling weight of KPWr |
| `NANO_RE_ENGLISH_RELATION_WEIGHT` | `1.0` | Sampling weight of Re-DocRED |
| `NANO_RE_MAX_RELATIONS` | unset | Cap the relation inventory to the most frequent |
| `NANO_RE_TRIM_VOCABULARY` | `false` | Compact the embedding table, see below |
| `NANO_RE_RELATION_LOSS` | `adaptive_threshold` | Or `bce` |

### Vocabulary trimming

The embedding table is most of the model: 98.3M of mmBERT-small's 140.5M
parameters. Trimming it to the tokens your languages actually use cuts the model
by roughly a factor of four.

It is off by default, because a released multilingual model should work in the
languages it advertises, and trimming silently degrades every language left out.
Turn it on when you are deploying to a fixed, known language set:

```bash
NANO_RE_TRIM_VOCABULARY=true NANO_RE_LANGUAGES=pl,en uv run nano-re all
```

If you do, sample your own documents into the token count as well. Rare
surnames, company names and domain abbreviations that appear in your text but
not in Wikipedia will otherwise be dropped and fall back to the unknown token.

## Tests

```bash
uv run pytest
```

The suite covers the places where a mistake produces no error rather than a
crash: identifier checksums, BIO span decoding at boundaries, mention
clustering, character offset alignment, corpus interleaving ratios, task masking
in the loss, and the vocabulary remap.

## Limitations

These are the things worth knowing before you build on it.

- **Relation quality is bounded by the training data.** SREDFM is generated
  automatically, so its labels are noisy and incomplete. Re-DocRED is
  human-corrected but English only. There is no gold relation evaluation set for
  Polish, in this project or anywhere else, so Polish relation quality can be
  estimated but not measured.
- **Relations are encyclopaedic.** The corpora describe the world as Wikidata
  does. Business specific relations such as "party to this contract" or "invoice
  issued by" are not among them, and no public corpus contains them.
- **Coreference is heuristic.** See above. Relation quality depends directly on
  it, and errors compound.
- **Entities are not linked to a knowledge base.** The model returns typed
  mentions, not Wikidata identifiers. Assigning canonical identity is entity
  linking, a separate and considerably larger system.
- **Entities are not merged across documents.** Clustering works within a single
  input. A corpus level entity registry belongs to whatever consumes this
  output.
- **The rule layer is tuned for Polish and EU identifiers.** Rules are plain
  data in `patterns/library.py`; adding your own numbering scheme means
  appending to a list, not editing extraction code.

## Acknowledgements

The corpora are the work of others and the citations are in the generated model
card. SREDFM and REDFM are from Babelscape, Re-DocRED from Tsinghua and
collaborators, KPWr from CLARIN-PL, and the encoder from the Johns Hopkins CLSP.
