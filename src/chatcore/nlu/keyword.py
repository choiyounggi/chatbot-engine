"""Core-morpheme exact match (opt-in per bot, Settings.keyword_match; builder plan §2.2).

Kiwi-tokenize the input, drop particles (J*), endings (E*), affixes (XS*, XP*)
and punctuation/symbols (S* except SL foreign words and SN numbers). Verbs and
adjectives come out as their stem (Kiwi already lemmatizes: "알려줬어요" ->
알리/VV + 어/EC + 주/VX + 었/EP + 어요/EF, so dropping E* leaves 알리/주). If what
remains equals a registered example's own core-morpheme sequence, it counts as
an exact match. Uses a separate Kiwi instance from the entity extractor's
(chatcore.nlu.entities.EntityExtractor): this one needs no bot-specific user words.
"""

from __future__ import annotations

import threading

from kiwipiepy import Kiwi

_DROP_PREFIXES = ("J", "E", "XS", "XP")  # particles, endings, suffixes, prefixes
_KEEP_SYMBOLS = frozenset({"SL", "SN"})  # foreign words, numbers; every other S* tag is punctuation


def _keep(tag: str) -> bool:
    if tag.startswith("S"):
        return tag in _KEEP_SYMBOLS
    return not tag.startswith(_DROP_PREFIXES)


class KeywordMatcher:
    def __init__(self, kiwi: Kiwi | None = None) -> None:
        self.kiwi = kiwi or Kiwi()
        self._lock = threading.Lock()  # Kiwi instances are not documented as thread-safe

    def morphemes(self, text: str) -> tuple[str, ...]:
        """The core-morpheme sequence used for matching; empty when nothing but
        grammar morphemes and punctuation remains."""
        with self._lock:
            tokens = self.kiwi.tokenize(text)
        return tuple(t.form for t in tokens if _keep(t.tag))
