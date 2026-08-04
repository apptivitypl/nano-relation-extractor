"""Tests for BIO decoding and mention clustering.

These two steps stand between the model's logits and anything a consumer sees,
and both fail quietly: a decoder that drops a span or a clusterer that merges
two organisations produces a plausible-looking result that is simply wrong.
"""

from __future__ import annotations

import torch

from nano_re.inference.clusterer import SurfaceFormClusterer
from nano_re.inference.decoder import BioSpanDecoder
from nano_re.inference.results import PredictedMention
from nano_re.schema import LabelSchema

SCHEMA = LabelSchema.from_relation_info({"P17": "country"})


def _logits(tags: list[str]) -> torch.Tensor:
    """Build one-hot logits selecting the requested tag per position.

    Args:
        tags: Desired BIO tag per sub-word position.

    Returns:
        A ``[S, L]`` tensor whose argmax is the requested tag.
    """
    bio_to_id = SCHEMA.bio_to_id
    scores = torch.zeros(len(tags), SCHEMA.num_bio_labels)
    for position, tag in enumerate(tags):
        scores[position, bio_to_id[tag]] = 1.0
    return scores


def _alignment(count: int) -> dict[int, list[int]]:
    """Map each word onto a single sub-word position.

    Args:
        count: Number of words.

    Returns:
        A one-to-one word to sub-word alignment.
    """
    return {index: [index] for index in range(count)}


def test_decodes_a_single_span() -> None:
    """A B- followed by I- yields one mention covering both words."""
    words = ["Skai", "TV", "nadaje"]
    decoded = BioSpanDecoder(SCHEMA).decode(
        _logits(["B-ORG", "I-ORG", "O"]), _alignment(3), words
    )
    assert [(m.text, t) for m, t in decoded] == [("Skai TV", "ORG")]


def test_adjacent_spans_do_not_merge() -> None:
    """A new B- closes the previous span rather than extending it."""
    words = ["Kowalski", "Nowak"]
    decoded = BioSpanDecoder(SCHEMA).decode(
        _logits(["B-PER", "B-PER"]), _alignment(2), words
    )
    assert [m.text for m, _ in decoded] == ["Kowalski", "Nowak"]


def test_inside_tag_without_begin_opens_a_span() -> None:
    """A stray I- is read as the start of a mention rather than discarded."""
    words = ["Warszawa", "lezy"]
    decoded = BioSpanDecoder(SCHEMA).decode(
        _logits(["I-LOC", "O"]), _alignment(2), words
    )
    assert [(m.text, t) for m, t in decoded] == [("Warszawa", "LOC")]


def test_type_change_splits_a_span() -> None:
    """An I- of a different type starts a new mention."""
    words = ["Adam", "Krakow"]
    decoded = BioSpanDecoder(SCHEMA).decode(
        _logits(["B-PER", "I-LOC"]), _alignment(2), words
    )
    assert [(m.text, t) for m, t in decoded] == [("Adam", "PER"), ("Krakow", "LOC")]


def test_span_running_to_the_end_is_emitted() -> None:
    """A mention touching the final word is not lost."""
    words = ["stacja", "Skai", "TV"]
    decoded = BioSpanDecoder(SCHEMA).decode(
        _logits(["O", "B-ORG", "I-ORG"]), _alignment(3), words
    )
    assert [m.text for m, _ in decoded] == ["Skai TV"]


def test_words_beyond_truncation_are_ignored() -> None:
    """Words with no sub-word alignment contribute nothing."""
    words = ["Skai", "TV", "poza", "oknem"]
    decoded = BioSpanDecoder(SCHEMA).decode(
        _logits(["B-ORG", "I-ORG"]), _alignment(2), words
    )
    assert [m.text for m, _ in decoded] == ["Skai TV"]


def test_clusterer_merges_repeated_surface_forms() -> None:
    """The same name seen twice becomes one entity with two mentions."""
    mentions = [
        (PredictedMention("Skai TV", 0, 2), "ORG"),
        (PredictedMention("Skai TV", 10, 12), "ORG"),
    ]
    entities = SurfaceFormClusterer().cluster(mentions)
    assert len(entities) == 1
    assert entities[0].mention_count == 2


def test_clusterer_merges_a_whole_word_prefix() -> None:
    """A shorter form joins its longer version and the longer name wins."""
    mentions = [
        (PredictedMention("Skai", 0, 1), "ORG"),
        (PredictedMention("Skai TV", 5, 7), "ORG"),
    ]
    entities = SurfaceFormClusterer().cluster(mentions)
    assert len(entities) == 1
    assert entities[0].name == "Skai TV"


def test_clusterer_keeps_types_apart() -> None:
    """Identical surface forms of different types stay separate."""
    mentions = [
        (PredictedMention("Kowalski", 0, 1), "PER"),
        (PredictedMention("Kowalski", 5, 6), "ORG"),
    ]
    assert len(SurfaceFormClusterer().cluster(mentions)) == 2


def test_clusterer_does_not_merge_on_a_shared_suffix() -> None:
    """Only prefixes merge, so unrelated names sharing a tail stay apart."""
    mentions = [
        (PredictedMention("Bank Handlowy", 0, 2), "ORG"),
        (PredictedMention("Handlowy", 5, 6), "ORG"),
    ]
    assert len(SurfaceFormClusterer().cluster(mentions)) == 2


def test_span_scoring_counts_boundaries_and_type_together() -> None:
    """A mention is correct only when both its extent and its type match."""
    from nano_re.training.metrics import decode_bio_spans

    assert decode_bio_spans(["B-PER", "I-PER", "O"]) == {(0, 2, "PER")}
    assert decode_bio_spans(["B-PER", "O", "O"]) != decode_bio_spans(
        ["B-PER", "I-PER", "O"]
    )
    assert decode_bio_spans(["B-ORG", "I-ORG"]) != decode_bio_spans(
        ["B-PER", "I-PER"]
    )


def test_span_scoring_splits_adjacent_mentions() -> None:
    """Two neighbouring mentions of one type are two spans, not one."""
    from nano_re.training.metrics import decode_bio_spans

    assert decode_bio_spans(["B-PER", "B-PER"]) == {(0, 1, "PER"), (1, 2, "PER")}


def test_span_scoring_reads_a_stray_inside_tag_as_a_start() -> None:
    """An I- without a B- opens a span rather than being dropped."""
    from nano_re.training.metrics import decode_bio_spans

    assert decode_bio_spans(["I-LOC", "O"]) == {(0, 1, "LOC")}


def test_span_scoring_closes_a_span_at_the_end_of_the_sequence() -> None:
    """A mention touching the final token is not lost."""
    from nano_re.training.metrics import decode_bio_spans

    assert decode_bio_spans(["O", "B-ORG", "I-ORG"]) == {(1, 3, "ORG")}


def test_span_scoring_of_an_empty_sequence_is_empty() -> None:
    """Nothing in, nothing out."""
    from nano_re.training.metrics import decode_bio_spans

    assert decode_bio_spans([]) == set()
