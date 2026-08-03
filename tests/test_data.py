"""Tests for corpus parsing, interleaving and task masking.

The parser translates character offsets onto words, which is where a corpus
change breaks silently. The task mask decides whether an entity-only corpus is
allowed to train the relation head, which is where a wrong answer degrades the
model without any error surfacing.
"""

from __future__ import annotations

import torch

from nano_re.data.collator import MultiTaskCollator
from nano_re.data.encoder import EncodedDocument
from nano_re.data.multi_corpus import MultiCorpusDataset
from nano_re.data.parsers import MultiNerdParser, SredfmParser
from nano_re.schema import RelationInventory

SREDFM_RECORD = {
    "docid": "1",
    "text": "Skai TV nadaje z Pireusu w Grecji.",
    "lan": "pl",
    "entities": [
        {"uri": "Q1", "boundaries": [0, 7], "surfaceform": "Skai TV", "type": "ORG"},
        {"uri": "Q2", "boundaries": [17, 24], "surfaceform": "Pireusu", "type": "LOC"},
    ],
    "relations": [
        {
            "subject": {"uri": "Q1", "boundaries": [0, 7], "surfaceform": "Skai TV"},
            "predicate": {"uri": "P17", "surfaceform": "panstwo"},
            "object": {"uri": "Q3", "boundaries": [27, 33], "surfaceform": "Grecji"},
        }
    ],
}


def test_sredfm_character_offsets_land_on_the_right_words() -> None:
    """Annotated spans resolve to the words carrying that surface form."""
    document = SredfmParser().parse(SREDFM_RECORD, 0)
    by_name = {
        entity.mentions[0].text: entity.mentions[0] for entity in document.entities
    }
    for name, mention in by_name.items():
        sliced = " ".join(document.words[mention.start : mention.end])
        assert sliced.replace(" ", "") == name.replace(" ", "")


def test_sredfm_relation_endpoints_resolve_to_entities() -> None:
    """A relation names entities the document actually contains."""
    document = SredfmParser().parse(SREDFM_RECORD, 0)
    assert len(document.relations) == 1
    triple = document.relations[0]
    assert document.entities[triple.head].mentions[0].text == "Skai TV"
    assert document.entities[triple.tail].mentions[0].text == "Grecji"
    assert triple.relation == "P17"


def test_sredfm_relation_arguments_become_entities() -> None:
    """An argument absent from the entity list is still registered."""
    document = SredfmParser().parse(SREDFM_RECORD, 0)
    assert document.num_entities == 3


def test_sredfm_populates_the_relation_inventory() -> None:
    """Parsing records the predicates that will size the relation head."""
    inventory = RelationInventory()
    SredfmParser(inventory=inventory).parse(SREDFM_RECORD, 0)
    assert inventory.counts == {"P17": 1}
    assert inventory.to_schema().describe_relation("P17") == "panstwo"


def test_sredfm_widens_a_mid_word_boundary() -> None:
    """An annotation ending inside a word still selects that whole word."""
    record = dict(SREDFM_RECORD)
    record["entities"] = [
        {"uri": "Q9", "boundaries": [0, 3], "surfaceform": "Ska", "type": "ORG"}
    ]
    record["relations"] = []
    document = SredfmParser().parse(record, 0)
    mention = document.entities[0].mentions[0]
    assert document.words[mention.start : mention.end] == ("Skai",)


def test_multinerd_reconstructs_spans_and_folds_types() -> None:
    """Tag indices become spans, and fine types fold onto the canonical set."""
    record = {"tokens": ["Adam", "Nowak", "je", "jablko"], "ner_tags": [1, 2, 0, 25],
              "lang": "pl"}
    document = MultiNerdParser().parse(record, 0)
    assert [(e.mentions[0].text, e.entity_type) for e in document.entities] == [
        ("Adam Nowak", "PER"),
        ("jablko", "MISC"),
    ]


def test_multinerd_documents_declare_no_relation_supervision() -> None:
    """An entity-only corpus must not claim relation labels."""
    record = {"tokens": ["Adam"], "ner_tags": [1], "lang": "pl"}
    document = MultiNerdParser().parse(record, 0)
    assert document.has_labels is False
    assert document.relations == ()


def _encoded(has_relations: bool) -> EncodedDocument:
    """Build a minimal encoded document.

    Args:
        has_relations: Whether the source corpus supervises relations.

    Returns:
        An encoded document with two entities and one pair.
    """
    return EncodedDocument(
        doc_id="d",
        input_ids=torch.tensor([1, 2, 3]),
        attention_mask=torch.tensor([1, 1, 1]),
        ner_labels=torch.tensor([0, 0, 0]),
        mention_mask=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        pair_index=torch.tensor([[0, 1]]),
        relation_labels=torch.tensor([[1.0, 0.0]]),
        entity_types=("ORG", "LOC"),
        source_entity_ids=(0, 1),
        has_relation_supervision=has_relations,
    )


def test_entity_only_documents_are_masked_out_of_the_relation_task() -> None:
    """A document without relation labels contributes no active pair."""
    batch = MultiTaskCollator(pad_token_id=0)([_encoded(False)])
    assert float(batch.pair_mask.sum()) == 0.0


def test_relation_documents_keep_their_pairs_active() -> None:
    """A document with relation labels contributes its pairs."""
    batch = MultiTaskCollator(pad_token_id=0)([_encoded(True)])
    assert float(batch.pair_mask.sum()) == 1.0


def test_collator_drops_unusable_documents() -> None:
    """A batch whose members are all unusable collapses to nothing."""
    assert MultiTaskCollator(pad_token_id=0)([None, None]) is None


def test_collator_keeps_the_usable_members_of_a_mixed_batch() -> None:
    """One usable document survives alongside unusable ones."""
    batch = MultiTaskCollator(pad_token_id=0)([None, _encoded(True), None])
    assert batch is not None
    assert batch.input_ids.shape[0] == 1


def _interleave(sizes: tuple[int, int], weights: list[float]) -> str:
    """Interleave two labelled corpora and return the emission order.

    Args:
        sizes: Number of documents in each corpus.
        weights: Sampling weight per corpus.

    Returns:
        A string of corpus labels in emission order.
    """
    dataset = MultiCorpusDataset.__new__(MultiCorpusDataset)
    corpora = [[f"a{i}" for i in range(sizes[0])], [f"b{i}" for i in range(sizes[1])]]
    return "".join(item[0] for item in dataset._interleave(corpora, weights))


def test_equal_weights_alternate_corpora() -> None:
    """Equal weights emit one document from each corpus in turn."""
    assert _interleave((3, 3), [1, 1]).startswith("abab")


def test_weights_control_the_emission_ratio() -> None:
    """A corpus weighted four times higher appears four times as often."""
    order = _interleave((10, 40), [1, 4])
    assert order.count("b") == 40 and order.count("a") == 10
    assert order[:5].count("b") == 4


def test_every_document_is_emitted_exactly_once() -> None:
    """Interleaving reorders documents without dropping or repeating them."""
    order = _interleave((7, 11), [1, 3])
    assert len(order) == 18


def test_zero_weight_drops_a_corpus_without_hanging() -> None:
    """A corpus weighted zero contributes nothing and does not stall."""
    assert _interleave((5, 5), [1, 0]) == "aaaaa"


def test_empty_corpora_yield_nothing() -> None:
    """Interleaving nothing returns nothing."""
    assert _interleave((0, 0), [1, 1]) == ""
