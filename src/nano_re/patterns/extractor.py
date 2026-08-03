"""Deterministic extraction of structured identifiers.

The neural model handles the entities a corpus can teach it: people,
organisations, places, dates. It cannot handle invoice numbers, tax
identifiers or bank accounts, because no NLP corpus contains them. Those have
fixed shapes and, usually, checksums, which makes them a matching problem rather
than a learning problem.

Matches are reported as ordinary entities so a consumer sees one uniform result,
but they are kept out of the relation head: it has no training signal for these
types and would only invent edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .library import POLISH_BUSINESS_RULES
from .rules import PatternRule

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class PatternMatch:
    """One identifier found by a rule.

    Attributes:
        rule: Name of the rule that matched.
        entity_type: Type assigned to the match.
        text: Surface form as it appears in the document.
        value: Canonical, normalised value.
        char_start: Inclusive character offset.
        char_end: Exclusive character offset.
        word_start: Inclusive word index, once aligned to a tokenisation.
        word_end: Exclusive word index.
    """

    rule: str
    entity_type: str
    text: str
    value: str
    char_start: int
    char_end: int
    word_start: int = -1
    word_end: int = -1


class PatternExtractor:
    """Applies a rule set to raw text and resolves overlapping matches.

    Args:
        rules: Rules to apply. Defaults to the Polish business rule set.
    """

    def __init__(self, rules: tuple[PatternRule, ...] = POLISH_BUSINESS_RULES) -> None:
        self._rules = rules

    @property
    def rules(self) -> tuple[PatternRule, ...]:
        """Rules this extractor applies."""
        return self._rules

    def extract(self, text: str) -> list[PatternMatch]:
        """Find every identifier in a text.

        Args:
            text: Raw input text.

        Returns:
            Matches in document order, with overlaps resolved.
        """
        candidates: list[PatternMatch] = []
        for rule in self._rules:
            for match in rule.pattern.finditer(text):
                span = self._selected_span(match)
                if span is None:
                    continue
                start, end, raw = span
                value = rule.normalise(raw)
                if not value or not rule.accepts(value):
                    continue
                candidates.append(
                    PatternMatch(
                        rule=rule.name,
                        entity_type=rule.entity_type,
                        text=raw,
                        value=value,
                        char_start=start,
                        char_end=end,
                    )
                )
        return self._resolve_overlaps(candidates)

    def align(self, matches: list[PatternMatch], text: str) -> list[PatternMatch]:
        """Attach word indices to matches for a given tokenisation.

        Args:
            matches: Matches produced by :meth:`extract`.
            text: The text the matches were found in.

        Returns:
            Matches carrying word ranges. Matches that align to no word are
            dropped, since a consumer cannot locate them.
        """
        spans = [
            (item.start(), item.end()) for item in TOKEN_PATTERN.finditer(text)
        ]
        aligned: list[PatternMatch] = []
        for match in matches:
            covered = [
                position
                for position, (start, end) in enumerate(spans)
                if start < match.char_end and end > match.char_start
            ]
            if not covered:
                continue
            aligned.append(
                PatternMatch(
                    rule=match.rule,
                    entity_type=match.entity_type,
                    text=match.text,
                    value=match.value,
                    char_start=match.char_start,
                    char_end=match.char_end,
                    word_start=covered[0],
                    word_end=covered[-1] + 1,
                )
            )
        return aligned

    def _selected_span(self, match: re.Match[str]) -> tuple[int, int, str] | None:
        """Return the span a rule intends to report.

        A rule may match surrounding context, such as the word ``NIP`` before
        the number, while only the ``value`` group should be reported.

        Args:
            match: A regular expression match.

        Returns:
            Start, end and text of the reported span, or ``None`` when empty.
        """
        if "value" in match.re.groupindex:
            start, end = match.span("value")
        else:
            start, end = match.span()
        if start < 0 or end <= start:
            return None
        return start, end, match.string[start:end]

    def _resolve_overlaps(self, matches: list[PatternMatch]) -> list[PatternMatch]:
        """Keep the best match where several rules cover the same characters.

        A bank account is also a run of digits and an IBAN body is also a
        document number; without arbitration the same characters would be
        reported several times under different types. Higher rule priority wins,
        then the longer span.

        Args:
            matches: Unresolved candidates.

        Returns:
            Non-overlapping matches in document order.
        """
        priority = {rule.name: rule.priority for rule in self._rules}
        ordered = sorted(
            matches,
            key=lambda item: (
                -priority.get(item.rule, 0),
                -(item.char_end - item.char_start),
                item.char_start,
            ),
        )
        kept: list[PatternMatch] = []
        for candidate in ordered:
            if any(
                candidate.char_start < existing.char_end
                and candidate.char_end > existing.char_start
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return sorted(kept, key=lambda item: item.char_start)

