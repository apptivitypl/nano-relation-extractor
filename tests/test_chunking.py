"""Tests for windowing long documents and merging the results.

Windows are sized against a measured sub-word budget rather than a word count,
because how many words fit varies by more than a factor of two between English
prose and a Polish invoice. Merging then has to undo the split without leaving
an entity duplicated across the boundary it straddled.
"""

from __future__ import annotations

from nano_re.inference.chunking import ResultMerger, TextChunker, Window
from nano_re.inference.results import (
    ExtractionResult,
    PredictedEntity,
    PredictedMention,
    PredictedRelation,
)


class FakeTokenizer:
    """A tokenizer whose sub-word cost is the length of the word.

    Using a stand-in keeps these tests independent of any downloaded model while
    still exercising the packing arithmetic, which is what can be wrong.
    """

    def __call__(self, words, is_split_into_words=False, add_special_tokens=True):
        """Return one identifier per character of each word.

        Args:
            words: Words to encode.
            is_split_into_words: Accepted for signature compatibility.
            add_special_tokens: Accepted for signature compatibility.

        Returns:
            A mapping with an ``input_ids`` list.
        """
        return {"input_ids": [0] * sum(len(word) for word in words)}


def _chunker(budget: int, overlap: float = 0.25) -> TextChunker:
    """Build a chunker over the stand-in tokenizer.

    Args:
        budget: Sub-word budget including reserved positions.
        overlap: Fraction of a window repeated in the next.

    Returns:
        The configured chunker.
    """
    return TextChunker(
        FakeTokenizer(), max_sequence_length=budget, overlap=overlap, reserved_tokens=0
    )


def test_short_document_yields_one_window() -> None:
    """A document inside the budget is not split."""
    words = ["aa", "bb", "cc"]
    assert _chunker(100).split(words) == [Window(0, 3)]


def test_long_document_is_split_and_fully_covered() -> None:
    """Every word appears in at least one window."""
    words = [f"w{index}" for index in range(40)]
    windows = _chunker(20).split(words)
    assert len(windows) > 1
    covered = {position for window in windows for position in range(window.start, window.end)}
    assert covered == set(range(40))


def test_windows_overlap() -> None:
    """Consecutive windows share words so boundary entities stay whole."""
    words = [f"w{index}" for index in range(40)]
    windows = _chunker(20, overlap=0.25).split(words)
    assert windows[1].start < windows[0].end


def test_window_respects_the_subword_budget() -> None:
    """No window costs more sub-words than the budget allows."""
    words = ["aaaa"] * 20
    windows = _chunker(12).split(words)
    for window in windows:
        assert sum(len(word) for word in words[window.start : window.end]) <= 12


def test_a_single_oversized_word_still_forms_a_window() -> None:
    """A word larger than the budget does not stall the packer."""
    windows = _chunker(4).split(["aaaaaaaaaa", "bb"])
    assert windows[0] == Window(0, 1)


def test_empty_document_yields_no_windows() -> None:
    """Nothing in, nothing out."""
    assert _chunker(10).split([]) == []


def _result(names: list[tuple[str, str, int, int]]) -> ExtractionResult:
    """Build a result holding one entity per supplied tuple.

    Args:
        names: Tuples of name, type, mention start and mention end.

    Returns:
        The assembled result.
    """
    entities = tuple(
        PredictedEntity(
            index=index,
            name=name,
            entity_type=entity_type,
            mentions=(PredictedMention(name, start, end),),
        )
        for index, (name, entity_type, start, end) in enumerate(names)
    )
    return ExtractionResult(words=(), entities=entities, relations=())


def test_merge_unifies_an_entity_seen_in_two_windows() -> None:
    """The same entity in overlapping windows becomes one entity."""
    merged = ResultMerger().merge(
        [
            (Window(0, 10), _result([("Skai TV", "ORG", 0, 2)])),
            (Window(8, 18), _result([("Skai TV", "ORG", 1, 3)])),
        ],
        words=tuple(f"w{index}" for index in range(18)),
    )
    assert len(merged.entities) == 1
    assert merged.entities[0].mention_count == 2


def test_merge_shifts_mentions_into_document_coordinates() -> None:
    """Window-local offsets become document-global."""
    merged = ResultMerger().merge(
        [(Window(8, 18), _result([("Nowak", "PER", 1, 2)]))],
        words=tuple(f"w{index}" for index in range(18)),
    )
    assert merged.entities[0].mentions[0].start == 9


def test_merge_keeps_distinct_entities_apart() -> None:
    """Different names remain different entities."""
    merged = ResultMerger().merge(
        [
            (Window(0, 5), _result([("Kowalski", "PER", 0, 1)])),
            (Window(0, 5), _result([("Nowak", "PER", 2, 3)])),
        ],
        words=tuple(f"w{index}" for index in range(5)),
    )
    assert len(merged.entities) == 2


def test_merge_deduplicates_relations_keeping_the_best() -> None:
    """A relation seen twice is reported once, at its highest confidence."""
    first = _result([("A", "ORG", 0, 1), ("B", "LOC", 2, 3)])
    first = ExtractionResult(
        words=(),
        entities=first.entities,
        relations=(PredictedRelation(0, 1, "P17", "country", 0.6),),
    )
    second = _result([("A", "ORG", 0, 1), ("B", "LOC", 2, 3)])
    second = ExtractionResult(
        words=(),
        entities=second.entities,
        relations=(PredictedRelation(0, 1, "P17", "country", 0.9),),
    )
    merged = ResultMerger().merge(
        [(Window(0, 5), first), (Window(0, 5), second)],
        words=tuple(f"w{index}" for index in range(5)),
    )
    assert len(merged.relations) == 1
    assert merged.relations[0].confidence == 0.9


def test_merge_of_nothing_is_empty() -> None:
    """Merging no windows yields an empty result."""
    merged = ResultMerger().merge([], words=("a",))
    assert merged.entities == () and merged.relations == ()
