"""Grouping mentions into entities.

The relation corpus supplies gold entity clusters, and the relation head is
trained to consume them. Raw text has none, so inference has to produce them. This module
is where that gap is filled, and it is filled by a heuristic rather than by a
coreference model: the pipeline trains no coreference component, and pretending
otherwise would hide the weakest link in end-to-end extraction.

The heuristic exploits the fact that the training corpora are encyclopaedic
text, where the same entity is usually repeated verbatim or as a prefix of its
full name.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from .results import PredictedEntity, PredictedMention

NORMALISE_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
STOP_PREFIXES = ("the ", "a ", "an ")


@runtime_checkable
class MentionClusterer(Protocol):
    """Groups typed mentions into entity clusters."""

    def cluster(
        self, mentions: list[tuple[PredictedMention, str]]
    ) -> list[PredictedEntity]:
        """Group mentions that refer to the same entity.

        Args:
            mentions: Pairs of mention and entity type, in reading order.

        Returns:
            The entity clusters, indexed in order of first appearance.
        """
        ...


class SurfaceFormClusterer:
    """Clusters mentions by normalised surface form and type.

    Two mentions join the same cluster when they share a type and their
    normalised forms are equal, or when one is a whole-word prefix of the other.
    The prefix rule links "Skai" to "Skai TV" and "Barack Obama" to "Obama"
    without pulling in unrelated names that merely share a substring.

    Args:
        match_prefixes: Whether to apply the prefix rule in addition to exact
            matching.
        min_prefix_words: Shortest prefix, in words, allowed to trigger a match.
            A single common word is too weak a signal on its own.
    """

    def __init__(self, match_prefixes: bool = True, min_prefix_words: int = 1) -> None:
        self._match_prefixes = match_prefixes
        self._min_prefix_words = min_prefix_words

    def cluster(
        self, mentions: list[tuple[PredictedMention, str]]
    ) -> list[PredictedEntity]:
        """Group mentions that refer to the same entity.

        Args:
            mentions: Pairs of mention and entity type, in reading order.

        Returns:
            The entity clusters, indexed in order of first appearance.
        """
        groups: list[dict[str, object]] = []
        for mention, entity_type in mentions:
            key = self._normalise(mention.text)
            if not key:
                continue
            target = self._find_group(groups, key, entity_type)
            if target is None:
                groups.append(
                    {
                        "type": entity_type,
                        "keys": {key},
                        "mentions": [mention],
                    }
                )
            else:
                keys = target["keys"]
                assert isinstance(keys, set)
                keys.add(key)
                members = target["mentions"]
                assert isinstance(members, list)
                members.append(mention)

        entities: list[PredictedEntity] = []
        for index, group in enumerate(groups):
            members = group["mentions"]
            assert isinstance(members, list)
            longest = max(members, key=lambda item: len(item.text))
            entities.append(
                PredictedEntity(
                    index=index,
                    name=longest.text,
                    entity_type=str(group["type"]),
                    mentions=tuple(members),
                )
            )
        return entities

    def _find_group(
        self, groups: list[dict[str, object]], key: str, entity_type: str
    ) -> dict[str, object] | None:
        """Locate the cluster a mention belongs to.

        Args:
            groups: Clusters built so far.
            key: Normalised surface form of the incoming mention.
            entity_type: Type of the incoming mention.

        Returns:
            The matching cluster, or ``None`` when the mention starts a new one.
        """
        for group in groups:
            if group["type"] != entity_type:
                continue
            keys = group["keys"]
            assert isinstance(keys, set)
            if key in keys:
                return group
            if self._match_prefixes and any(
                self._is_word_prefix(key, other) or self._is_word_prefix(other, key)
                for other in keys
            ):
                return group
        return None

    def _is_word_prefix(self, shorter: str, longer: str) -> bool:
        """Test whether one form is a whole-word prefix of another.

        Args:
            shorter: Candidate prefix.
            longer: Candidate full form.

        Returns:
            ``True`` when ``shorter`` begins ``longer`` on a word boundary.
        """
        if shorter == longer or len(shorter) >= len(longer):
            return False
        shorter_words = shorter.split()
        longer_words = longer.split()
        if len(shorter_words) < self._min_prefix_words:
            return False
        return longer_words[: len(shorter_words)] == shorter_words

    def _normalise(self, text: str) -> str:
        """Reduce a surface form to a comparable key.

        Args:
            text: Raw surface form.

        Returns:
            Lowercased text without punctuation or leading articles.
        """
        cleaned = NORMALISE_PATTERN.sub(" ", text.lower())
        collapsed = " ".join(cleaned.split())
        for prefix in STOP_PREFIXES:
            if collapsed.startswith(prefix):
                return collapsed[len(prefix) :]
        return collapsed
