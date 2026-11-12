"""Typo correction against a bot's own vocabulary (docs/engine-core-plan.md "빌더 단계 E-I").

A sentence embedding reads "날시" as a different word from "날씨", so a block or an FAQ is missed by one
mistyped jamo. Before a turn is understood, each Hangul word the bot has never used is compared, jamo by
jamo, with the words it has (block examples, entity values, FAQ, documents): when a known word is one
jamo away (a jamo replaced, added or dropped; of several, the one the bot uses most) AND a Korean
language model (Kiwi) finds that word clearly more likely than what was typed, it is used instead.
Words the bot knows, short words (one syllable), real words the bot just never used ("먼지", "뭐임") and
candidates used equally often are left alone.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

_BASE, _JUNG, _JONG = 0xAC00, 21, 28
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNGS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONGS = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_HANGUL = re.compile(r"[가-힣]+")
MIN_SYLLABLES = 2
MAX_SUFFIX = 3  # syllables a word may carry after a known stem (particles, endings): "날시가요" -> "날시"
MARGIN = 5.0  # how much more likely (Kiwi log score) the known word must be (bench/spelling_eval.py)
MAX_SCORES = 20_000  # remembered Kiwi scores per bot version; users type new words forever, so it starts over


def jamo(word: str) -> str:
    """Hangul syllables as initial/medial/final jamo; a missing final is "·" so positions line up."""
    out = []
    for c in word:
        code = ord(c) - _BASE
        if 0 <= code < 11172:
            jong = code % _JONG
            cho, jung = _CHO[code // (_JUNG * _JONG)], _JUNGS[(code % (_JUNG * _JONG)) // _JONG]
            out += [cho, jung, _JONGS[jong] if jong else "·"]
        else:
            out.append(c)
    return "".join(out)


def _deletes(s: str) -> set[str]:
    return {s[:i] + s[i + 1:] for i in range(len(s))}


def _one_edit(a: str, b: str) -> bool:
    """Levenshtein distance exactly 1."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


@dataclass(frozen=True)
class Correction:
    text: str  # the sentence to understand
    changes: tuple[tuple[str, str], ...]  # (as typed, as the bot knows it)


Score = Callable[[str], float]  # a word -> how likely it is as Korean (higher = more likely)


def kiwi_score(kiwi: object) -> Score:
    """A Score from a kiwipiepy.Kiwi: the log score of the word's best analysis."""
    def score(word: str) -> float:
        return float(kiwi.analyze(word, top_n=1)[0][1])  # type: ignore[attr-defined]
    return score


_SHARED: dict[str, Any] = {}
_SHARED_LOCK = threading.Lock()


def shared_score() -> Score:
    """kiwi_score of one Kiwi for the whole process, created on first use and used under a lock (a
    bot's entity extractor has its own, with its user words; scoring needs none)."""
    def score(word: str) -> float:
        with _SHARED_LOCK:
            if "kiwi" not in _SHARED:
                from kiwipiepy import Kiwi

                _SHARED["kiwi"] = Kiwi()
            return float(_SHARED["kiwi"].analyze(word, top_n=1)[0][1])
    return score


class Speller:
    def __init__(self, texts: Iterable[str], score: Score | None = None, margin: float = MARGIN) -> None:
        """`texts`: everything the bot says it understands (examples, entity values, FAQ, documents).
        `score`: a Korean language model; without one every one-jamo match is taken (tests only)."""
        self.score, self.margin = score, margin
        self._scores: dict[str, float] = {}
        counts: Counter[str] = Counter()
        for text in texts:
            for word in _HANGUL.findall(text):
                for end in range(MIN_SYLLABLES, len(word) + 1):  # the word and its stems ("날씨가" -> "날씨")
                    counts[word[:end]] += 1
        self.known = counts
        self._by_delete: dict[str, set[str]] = {}
        for word in counts:
            j = jamo(word)
            for key in {j, *_deletes(j)}:
                self._by_delete.setdefault(key, set()).add(word)

    def __len__(self) -> int:
        return len(self.known)

    def _nearest(self, word: str, rest: str = "") -> str | None:
        """The one known word one jamo from `word` (the most used one if several share the best count),
        if `known + rest` reads clearly more likely than `word + rest` (the whole word as typed)."""
        j = jamo(word)
        found = {w for key in {j, *_deletes(j)} for w in self._by_delete.get(key, ()) if _one_edit(j, jamo(w))}
        found = {w for w in found if len(w) == len(word)}  # a jamo slip keeps the syllable count
        if not found:
            return None
        ranked = sorted(found, key=lambda w: -self.known[w])
        if len(ranked) > 1 and self.known[ranked[0]] == self.known[ranked[1]]:
            return None  # two equally likely words: do not guess
        best = ranked[0]
        if self.score is not None and self._likely(best + rest) - self._likely(word + rest) < self.margin:
            return None  # what was typed is a word in its own right
        return best

    def _likely(self, word: str) -> float:
        if word not in self._scores:
            assert self.score is not None
            if len(self._scores) >= MAX_SCORES:
                self._scores.clear()
            self._scores[word] = self.score(word)
        return self._scores[word]

    def correct(self, text: str) -> Correction:
        changes: list[tuple[str, str]] = []

        def fix(m: re.Match[str]) -> str:
            word = m.group(0)
            if len(word) < MIN_SYLLABLES or word in self.known:
                return word
            for cut in range(0, min(MAX_SUFFIX, len(word) - MIN_SYLLABLES) + 1):
                stem, rest = word[:len(word) - cut], word[len(word) - cut:]
                if stem in self.known:
                    return word  # a known stem with an ending the bot has not seen: not a typo
                near = self._nearest(stem, rest)
                if near is not None:
                    changes.append((stem, near))
                    return near + rest
            return word

        corrected = _HANGUL.sub(fix, text)
        return Correction(corrected, tuple(changes))


def vocabulary(bot: Any, kb: Any = None) -> list[str]:
    """What users of this bot are expected to write: block examples ({자리} left out), entity values and
    synonyms, and, with a knowledge base, its FAQ questions, similar questions, answers and document
    chunks. (The bot's own replies are not user words.)"""
    texts = [e for intent in bot.intents.values() for e in intent.examples]
    for entity in bot.entities.values():
        for canon, synonyms in (getattr(entity, "values", None) or {}).items():
            texts += [canon, *synonyms]
    if kb is not None:
        for e in kb.faq:
            texts += [e.title, *e.similar, e.body]
        texts += [c.body for c in kb.chunks]
    return texts
