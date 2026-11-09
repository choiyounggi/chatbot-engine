"""System entities: numbers, dates and times in everyday Korean.

Deliberately small: the common forms below, nothing clever. Anything else is
not recognized (the scenario then asks again), which is safer than guessing
a wrong value. All relative expressions are resolved against an injected date.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

NATIVE = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "넷": 4, "네": 4, "다섯": 5, "여섯": 6,
          "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "열한": 11, "열두": 12}
WEEKDAYS = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
RELATIVE_DAYS = {"그저께": -2, "그제": -2, "어제": -1, "오늘": 0, "내일": 1, "모레": 2, "글피": 3}
# What may follow a counter word inside a sentence ("세 개를", "두 번만"); anything else means the
# match is part of another word ("네 번호", "한 개미") and is not a number.
PARTICLES = r"(?:이서|에서|이|가|을|를|은|는|만|씩|째|도|에|의|로|요|정도|쯤|까지|부터|이나|나|만큼|짜리|뿐)"
NOT_ASCII_WORD = r"(?<![0-9A-Za-z.\-])"


@dataclass(frozen=True)
class Found:
    value: int | float | str
    start: int
    end: int


_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# A minus sign only at the start of a token, so "2026-11-10" and "010-1234" give no negatives.
NUMBER = re.compile(
    rf"(?:(?P<neg>영하\s*|{NOT_ASCII_WORD}-)(?P<signed>{_NUM})|{NOT_ASCII_WORD}(?P<num>{_NUM}))(?![0-9])"
)
NATIVE_COUNT = re.compile(
    r"(?<![가-힣])(?P<w>열[한두]|하나|다섯|여섯|일곱|여덟|아홉|[한두세네]|둘|셋|넷|열)\s*"
    rf"(?P<unit>개|명|번|살|가지|잔|권|장)(?=$|[^가-힣]|{PARTICLES}(?![가-힣])|{PARTICLES}{PARTICLES})"
)


def find_numbers(text: str) -> list[Found]:
    out: list[Found] = []
    for m in NUMBER.finditer(text):
        raw = (m.group("signed") or m.group("num")).replace(",", "")
        value: int | float = float(raw) if "." in raw else int(raw)
        if m.group("neg"):
            value = -abs(value)
        out.append(Found(value, m.start(), m.end()))
    for m in NATIVE_COUNT.finditer(text):
        if not any(f.start < m.end() and m.start() < f.end for f in out):
            out.append(Found(NATIVE[m.group("w")], m.start(), m.start() + len(m.group("w"))))
    return sorted(out, key=lambda f: f.start)


DATE_PATTERNS = [
    ("iso", re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")),
    ("ymd", re.compile(r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월\s*(?P<d>\d{1,2})\s*일")),
    ("md", re.compile(r"(?P<m>\d{1,2})\s*월\s*(?P<d>\d{1,2})\s*일")),
    ("after", re.compile(r"(?P<n>\d{1,3})\s*일\s*(?:후|뒤)")),
    ("week", re.compile(r"(?:(?P<which>이번|다음|담|저번|지난)\s*(?P<week>주)?\s*)?(?P<wd>[월화수목금토일])요일")),
    ("rel", re.compile("|".join(sorted(RELATIVE_DAYS, key=len, reverse=True)))),
]


def find_dates(text: str, today: dt.date) -> list[Found]:
    out: list[Found] = []
    for kind, pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            if any(f.start < m.end() and m.start() < f.end for f in out):
                continue  # a longer form already covered this span
            date = _resolve_date(kind, m, today)
            if date is not None:
                out.append(Found(date.isoformat(), m.start(), m.end()))
    return sorted(out, key=lambda f: f.start)


def _resolve_date(kind: str, m: re.Match[str], today: dt.date) -> dt.date | None:
    try:
        if kind in ("iso", "ymd"):
            return dt.date(int(m["y"]), int(m["m"]), int(m["d"]))
        if kind == "md":  # no year: the occurrence closest to today
            options = []
            for year in (today.year - 1, today.year, today.year + 1):
                try:
                    options.append(dt.date(year, int(m["m"]), int(m["d"])))
                except ValueError:
                    continue
            return min(options, key=lambda d: (abs((d - today).days), d)) if options else None
    except ValueError:
        return None  # 2월 30일 and the like
    if kind == "after":
        return today + dt.timedelta(days=int(m["n"]))
    if kind == "rel":
        return today + dt.timedelta(days=RELATIVE_DAYS[m.group(0)])
    target = WEEKDAYS[m["wd"]]
    which, week = m["which"], m["week"]
    monday = today - dt.timedelta(days=today.weekday())
    ahead = (target - today.weekday()) % 7
    if which == "이번":  # 이번 주 X요일 / 이번 X요일: this Mon-Sun week
        return monday + dt.timedelta(days=target)
    if which in ("다음", "담"):
        # 다음 주 X요일: next week's; 다음 X요일: the next one after today
        return monday + dt.timedelta(days=7 + target) if week else today + dt.timedelta(days=ahead or 7)
    if which in ("저번", "지난"):
        # 지난주 X요일: last week's; 지난 X요일: the most recent one before today
        return monday + dt.timedelta(days=target - 7) if week else today - dt.timedelta(days=(7 - ahead) % 7 or 7)
    return today + dt.timedelta(days=ahead)  # bare X요일: the next one, today included


TIME = re.compile(
    r"(?P<ap>오전|오후|아침|낮|저녁|밤|새벽)?\s*"
    r"(?:(?P<h>\d{1,2}):(?P<mi>\d{2})|(?P<h2>\d{1,2})\s*시(?!간)(?:\s*(?P<mi2>\d{1,2})\s*분|\s*(?P<half>반))?)"
    r"|(?P<word>정오|자정)"
)


def find_times(text: str) -> list[Found]:
    out: list[Found] = []
    for m in TIME.finditer(text):
        value = _resolve_time(m)
        if value is not None:
            start = m.start() + len(m.group(0)) - len(m.group(0).lstrip())
            out.append(Found(value, start, m.end()))
    return out


def _resolve_time(m: re.Match[str]) -> str | None:
    g = m.groupdict()
    if g.get("word"):
        return "12:00" if g["word"] == "정오" else "00:00"
    hour = int(g["h"] or g["h2"])
    minute = 30 if g.get("half") else int(g["mi"] or g["mi2"] or 0)
    ap = g.get("ap")
    if ap in ("오후", "저녁") and hour < 12:
        hour += 12
    elif ap == "밤":  # 밤 9시 = 21:00, 밤 12시 = 00:00, 밤 1시 = 01:00
        hour = 0 if hour == 12 else hour + 12 if 6 <= hour < 12 else hour
    elif ap == "낮" and hour < 6:
        hour += 12
    elif ap in ("오전", "새벽", "아침") and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"
