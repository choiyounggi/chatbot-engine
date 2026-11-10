"""Text normalization and entity extraction.

`EntityExtractor.extract` returns every candidate, possibly overlapping:
the interpreter knows which entity types matter right now (the pending slot,
the predicted scenario) and resolves overlaps among those only.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
import unicodedata
from dataclasses import dataclass

from kiwipiepy import Kiwi

from chatcore.bot.schema import BotDefinition, DictionaryEntity, RegexEntity
from chatcore.nlu import sysentities

PUNCT_EDGES = re.compile(r"^[\s.,!?~…·'\"()\[\]]+|[\s.,!?~…·'\"()\[\]]+$")


def normalize(text: str) -> str:
    """NFKC, collapse whitespace, trim."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def bare(text: str) -> str:
    """Normalized and without surrounding punctuation — for exact-match lookups."""
    return PUNCT_EDGES.sub("", normalize(text))


@dataclass(frozen=True)
class EntityMatch:
    entity: str  # entity ID or sys.number / sys.date / sys.time and the E9 sys.* system entities
    value: str | int | float  # canonical value
    text: str  # the matched surface text
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class EntityExtractor:
    """Dictionary entities on Kiwi token boundaries (so "서울" is not found
    inside "서울역"), regex entities, and system entities.

    Registers every dictionary value as a user word on the Kiwi instance, so
    pass a Kiwi that is not shared with another bot (the default creates one)."""

    def __init__(self, bot: BotDefinition, kiwi: Kiwi | None = None) -> None:
        self.bot = bot
        self.kiwi = kiwi or Kiwi()
        self._lock = threading.Lock()  # Kiwi instances are not documented as thread-safe
        self._forms: list[tuple[str, str, str]] = []  # (surface form, entity, canonical), longest first
        for eid, ent in bot.entities.items():
            if isinstance(ent, DictionaryEntity):
                for canon, synonyms in ent.values.items():
                    for form in [canon, *synonyms]:
                        self._forms.append((normalize(form), eid, canon))
                        if " " not in form:
                            self.kiwi.add_user_word(normalize(form), "NNP")
        self._forms.sort(key=lambda f: (-len(f[0]), f[0]))
        self._lookup = {f: (eid, canon) for f, eid, canon in self._forms}
        self._regex = {eid: re.compile(ent.pattern)
                       for eid, ent in bot.entities.items() if isinstance(ent, RegexEntity)}

    def extract(self, text: str, today: dt.date) -> list[EntityMatch]:
        text = normalize(text)
        if not text:
            return []
        out: list[EntityMatch] = []
        # Every occurrence of every form (overlaps included): if "서울시" fails the token
        # boundary check in "서울시립대", "서울" at the same position still gets its chance.
        hits = [(form, i) for form, _, _ in self._forms for i in _occurrences(text, form)]
        if hits:
            with self._lock:
                tokens = self.kiwi.tokenize(text)
            starts = {t.start for t in tokens}
            ends = {t.start + t.len for t in tokens}
            for form, i in hits:
                if i in starts and i + len(form) in ends:
                    eid, canon = self._lookup[form]
                    out.append(EntityMatch(eid, canon, form, i, i + len(form)))
        for eid, rx in self._regex.items():
            out += [EntityMatch(eid, m.group(0), m.group(0), m.start(), m.end())
                    for m in rx.finditer(text) if m.group(0)]
        system = (("sys.date", sysentities.find_dates(text, today)), ("sys.time", sysentities.find_times(text)),
                  ("sys.number", sysentities.find_numbers(text)),
                  ("sys.number.ordinal", sysentities.find_ordinals(text)),
                  ("sys.number.age", sysentities.find_ages(text)),
                  ("sys.number.percent", sysentities.find_percents(text)),
                  ("sys.date.period", sysentities.find_date_periods(text, today)),
                  ("sys.time.period", sysentities.find_time_periods(text)),
                  ("sys.url", sysentities.find_urls(text)),
                  ("sys.unit.currency", sysentities.find_currency(text)),
                  ("sys.unit.length", sysentities.find_length(text)),
                  ("sys.unit.weight", sysentities.find_weight(text)),
                  ("sys.unit.volume", sysentities.find_volume(text)),
                  ("sys.unit.temperature", sysentities.find_temperature(text)),
                  ("sys.unit.speed", sysentities.find_speed(text)))
        for entity, found in system:
            out += [EntityMatch(entity, f.value, text[f.start:f.end], f.start, f.end) for f in found]
        return sorted(out, key=lambda e: (e.start, -e.length, e.entity))


def _occurrences(text: str, form: str) -> list[int]:
    found, i = [], text.find(form)
    while i != -1:
        found.append(i)
        i = text.find(form, i + 1)
    return found


def pick(matches: list[EntityMatch], wanted: set[str]) -> list[EntityMatch]:
    """Matches of the wanted entity types, without overlaps (longer wins, then earlier)."""
    chosen: list[EntityMatch] = []
    for m in sorted((m for m in matches if m.entity in wanted), key=lambda m: (-m.length, m.start)):
        if all(m.end <= c.start or c.end <= m.start for c in chosen):
            chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)
