"""System entities: numbers, dates and times in everyday Korean.

Deliberately small: the common forms below, nothing clever. Anything else is
not recognized (the scenario then asks again), which is safer than guessing
a wrong value. All relative expressions are resolved against an injected date.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Callable
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


# ---------------------------------------------------------------------------
# E9 system entities: ordinals, ages, percents, periods, urls, units.
# ---------------------------------------------------------------------------

NEW_ENTITIES: frozenset[str] = frozenset({
    "sys.number.ordinal", "sys.number.age", "sys.number.percent",
    "sys.date.period", "sys.time.period", "sys.url",
    "sys.unit.currency", "sys.unit.length", "sys.unit.weight",
    "sys.unit.volume", "sys.unit.temperature", "sys.unit.speed",
    "sys.text",
})


# Numbers for the E9 finders: at most 15 digits (a float never overflows), never starting inside
# another number ("1,000" is not "000").
_BNUM = r"(?<![0-9,.])(?:\d{1,3}(?:,\d{3}){1,4}(?:\.\d+)?|\d{1,15}(?:\.\d+)?)(?![0-9])"
# What may follow a Korean unit word: the end, a non-Hangul character, or a particle / copula
# ("3원이에요", "30프로예요") — not more of a word ("3원칙", "3세트", "3프로젝트").
_KO_TAIL = rf"(?=$|[^가-힣]|{PARTICLES}|예|이|입|라|야)"


def _longest_first(cands: list[tuple[int, int, int | float | str]]) -> list[Found]:
    """From overlapping candidate spans, keep the longest ones (existing finders do this inline;
    the new ones combine several alternatives so a shared helper avoids repeating it each time)."""
    out: list[Found] = []
    for start, end, value in sorted(cands, key=lambda c: c[1] - c[0], reverse=True):
        if not any(f.start < end and start < f.end for f in out):
            out.append(Found(value, start, end))
    return sorted(out, key=lambda f: f.start)


def _to_number(s: str) -> float:
    return float(s.replace(",", ""))


def _fmt_num(n: float) -> int | float:
    return int(n) if n == int(n) else n


# --- sys.number.ordinal -----------------------------------------------------

ORDINAL_FIRST = re.compile(r"첫\s*(?:번\s*)?째")
ORDINAL_DIGIT = re.compile(rf"{NOT_ASCII_WORD}(?P<n>\d+)\s*번째")
ORDINAL_NATIVE_BEON = re.compile(
    r"(?<![가-힣])(?:(?P<tens>열|스물|스무|서른|마흔|쉰|예순|일흔|여든|아흔)(?P<ones>한|두|세|네|다섯|여섯|일곱|여덟|아홉)?"
    r"|(?P<w>두|세|네|다섯|여섯|일곱|여덟|아홉))\s*번째"
)
ORDINAL_JJAE = re.compile(r"(?<![가-힣])(?P<w>둘|셋|넷|다섯|여섯|일곱|여덟|아홉|열)째")
ORDINAL_NATIVE_MAP = {"두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
ORDINAL_JJAE_MAP = {"둘": 2, "셋": 3, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}


def find_ordinals(text: str) -> list[Found]:
    cands: list[tuple[int, int, int | float | str]] = []
    for m in ORDINAL_FIRST.finditer(text):
        cands.append((m.start(), m.end(), 1))
    for m in ORDINAL_DIGIT.finditer(text):
        n = int(m["n"])
        if n >= 1:
            cands.append((m.start(), m.end(), n))
    for m in ORDINAL_NATIVE_BEON.finditer(text):
        if m["tens"]:
            value = AGE_TENS[m["tens"]] + (AGE_ONES[m["ones"]] if m["ones"] else 0)
        else:
            value = ORDINAL_NATIVE_MAP[m["w"]]
        cands.append((m.start(), m.end(), value))
    for m in ORDINAL_JJAE.finditer(text):
        cands.append((m.start(), m.end(), ORDINAL_JJAE_MAP[m["w"]]))
    return _longest_first(cands)


# --- sys.number.age ----------------------------------------------------------

AGE_DIGIT = re.compile(rf"(?:만\s*)?{NOT_ASCII_WORD}(?P<n>\d+)\s*(?:살|세){_KO_TAIL}")
AGE_NATIVE_TENS_ONES = re.compile(
    r"(?P<tens>열|스물|서른|마흔|쉰|예순|일흔|여든|아흔)(?P<ones>한|두|세|네|다섯|여섯|일곱|여덟|아홉)\s*살"
)
AGE_NATIVE_TENS_ALONE = re.compile(r"(?P<tens>열|스물|스무|서른|마흔|쉰|예순|일흔|여든|아흔)\s*살")
AGE_NATIVE_ONES_ALONE = re.compile(r"(?P<ones>한|두|세|네|다섯|여섯|일곱|여덟|아홉)\s*살")
AGE_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50,
            "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
AGE_ONES = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}


def find_ages(text: str) -> list[Found]:
    cands: list[tuple[int, int, int | float | str]] = []
    for m in AGE_DIGIT.finditer(text):
        n = int(m["n"])
        if 0 <= n <= 150:
            cands.append((m.start(), m.end(), n))
    for m in AGE_NATIVE_TENS_ONES.finditer(text):
        cands.append((m.start(), m.end(), AGE_TENS[m["tens"]] + AGE_ONES[m["ones"]]))
    for m in AGE_NATIVE_TENS_ALONE.finditer(text):
        cands.append((m.start(), m.end(), AGE_TENS[m["tens"]]))
    for m in AGE_NATIVE_ONES_ALONE.finditer(text):
        cands.append((m.start(), m.end(), AGE_ONES[m["ones"]]))
    return _longest_first(cands)


# --- sys.number.percent -------------------------------------------------------

PERCENT = re.compile(rf"{NOT_ASCII_WORD}(?P<n>{_BNUM})\s*(?:%|(?:퍼센트|프로){_KO_TAIL})")


def find_percents(text: str) -> list[Found]:
    out = [Found(_fmt_num(_to_number(m["n"])), m.start(), m.end()) for m in PERCENT.finditer(text)]
    return sorted(out, key=lambda f: f.start)


# --- sys.date.period -----------------------------------------------------------

CONNECTOR = re.compile(r"\s*(?:부터|에서|~|-)\s*")
TRAILING_KKAJI = re.compile(r"\s*까지")
BARE_DAY = re.compile(r"(?P<d>\d{1,2})\s*일")

NAMED_PERIOD = [
    ("week_this", re.compile(r"이번\s*주(?!\s*말)(?!\s*[월화수목금토일]\s*요일)")),
    ("week_next", re.compile(r"(?:다음\s*주|담주)(?!\s*말)(?!\s*[월화수목금토일]\s*요일)")),
    ("week_last", re.compile(r"(?:지난주|저번\s*주)(?!\s*[월화수목금토일]\s*요일)")),
    ("month_this", re.compile(r"이번\s*달")),
    ("month_next", re.compile(r"다음\s*달")),
    ("month_last", re.compile(r"지난달|저번\s*달")),
    ("weekend_next", re.compile(r"다음\s*주말")),
    ("weekend_this", re.compile(r"이번\s*주말|주말")),
    ("year_this", re.compile(r"올해")),
    ("year_next", re.compile(r"내년")),
    ("year_last", re.compile(r"작년")),
]


def _month_range(year: int, month: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, month, 1)
    end = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return start, end


def _named_period_range(kind: str, today: dt.date) -> tuple[dt.date, dt.date]:
    monday = today - dt.timedelta(days=today.weekday())
    if kind == "week_this":
        return monday, monday + dt.timedelta(days=6)
    if kind == "week_next":
        return monday + dt.timedelta(days=7), monday + dt.timedelta(days=13)
    if kind == "week_last":
        return monday - dt.timedelta(days=7), monday - dt.timedelta(days=1)
    if kind == "month_this":
        return _month_range(today.year, today.month)
    if kind == "month_next":
        y, mo = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return _month_range(y, mo)
    if kind == "month_last":
        y, mo = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        return _month_range(y, mo)
    if kind == "weekend_this":
        return monday + dt.timedelta(days=5), monday + dt.timedelta(days=6)
    if kind == "weekend_next":
        return monday + dt.timedelta(days=12), monday + dt.timedelta(days=13)
    if kind == "year_this":
        return dt.date(today.year, 1, 1), dt.date(today.year, 12, 31)
    if kind == "year_next":
        return dt.date(today.year + 1, 1, 1), dt.date(today.year + 1, 12, 31)
    return dt.date(today.year - 1, 1, 1), dt.date(today.year - 1, 12, 31)  # year_last


def find_date_periods(text: str, today: dt.date) -> list[Found]:
    cands: list[tuple[int, int, int | float | str]] = []
    for kind, rx in NAMED_PERIOD:
        for m in rx.finditer(text):
            start_d, end_d = _named_period_range(kind, today)
            cands.append((m.start(), m.end(), f"{start_d.isoformat()}/{end_d.isoformat()}"))
    dates = find_dates(text, today)
    for d1 in dates:
        cm = CONNECTOR.match(text, d1.end)
        if not cm:
            continue
        rest_start = cm.end()
        start_date = dt.date.fromisoformat(str(d1.value))
        d2 = next((d for d in dates if d.start == rest_start), None)
        if d2 is not None:
            end_date = dt.date.fromisoformat(str(d2.value))
            end_pos = d2.end
        else:
            bm = BARE_DAY.match(text, rest_start)
            if not bm:
                continue
            try:
                end_date = start_date.replace(day=int(bm["d"]))
            except ValueError:
                continue
            end_pos = bm.end()
        if end_date < start_date:
            continue  # end before start: not recognized
        km = TRAILING_KKAJI.match(text, end_pos)
        final_end = km.end() if km else end_pos
        cands.append((d1.start, final_end, f"{start_date.isoformat()}/{end_date.isoformat()}"))
    return _longest_first(cands)


# --- sys.time.period -----------------------------------------------------------

def find_time_periods(text: str) -> list[Found]:
    cands: list[tuple[int, int, int | float | str]] = []
    for m1 in TIME.finditer(text):
        val1 = _resolve_time(m1)
        if val1 is None:
            continue
        cm = CONNECTOR.match(text, m1.end())
        if not cm:
            continue
        m2 = TIME.match(text, cm.end())
        if not m2:
            continue
        val2 = _resolve_time(m2)
        if val2 is None:
            continue
        ap1 = m1.group("ap")
        if ap1 in ("오후", "저녁", "밤") and not m2.group("ap") and val2 < val1:
            h2, mi2 = val2.split(":")
            later = f"{int(h2) + 12:02d}:{mi2}"
            if int(h2) < 12 and later > val1:  # "오후 2시부터 4시까지"; "밤 11시부터 2시까지" crosses midnight
                val2 = later
        start1 = m1.start() + len(m1.group(0)) - len(m1.group(0).lstrip())
        end2 = m2.end()
        km = TRAILING_KKAJI.match(text, end2)
        final_end = km.end() if km else end2
        cands.append((start1, final_end, f"{val1}/{val2}"))
    return _longest_first(cands)


# --- sys.url ---------------------------------------------------------------

URL = re.compile(r"(?:https?://|www\.)[^\s가-힣]+")
URL_TRAILING_PUNCT = ".,!?)]'\""


def find_urls(text: str) -> list[Found]:
    out: list[Found] = []
    for m in URL.finditer(text):
        raw, end = m.group(0), m.end()
        while raw and raw[-1] in URL_TRAILING_PUNCT:
            raw, end = raw[:-1], end - 1
        if not raw:
            continue
        value = raw if raw.startswith("http") else f"https://{raw}"
        out.append(Found(value, m.start(), end))
    return out


# --- sys.unit.currency -------------------------------------------------------

CURRENCY_MAN_CHEON = re.compile(rf"(?P<man>{_BNUM})\s*만\s*(?:(?P<cheon>{_BNUM})\s*천\s*)?원{_KO_TAIL}")
CURRENCY_CHEON_ONLY = re.compile(rf"(?P<n>{_BNUM})\s*천\s*원{_KO_TAIL}")
CURRENCY_EOK = re.compile(rf"(?P<n>{_BNUM})\s*억\s*원{_KO_TAIL}")
CURRENCY_PLAIN = re.compile(rf"(?P<n>{_BNUM})\s*(?P<unit>원|달러|엔|유로){_KO_TAIL}")
CURRENCY_DOLLAR = re.compile(rf"\$\s*(?P<n>{_BNUM})")
CURRENCY_CODE = {"원": "KRW", "달러": "USD", "엔": "JPY", "유로": "EUR"}


def _currency_str(n: float, code: str) -> str:
    return f"{_fmt_num(n)} {code}"


def find_currency(text: str) -> list[Found]:
    cands: list[tuple[int, int, int | float | str]] = []
    for m in CURRENCY_MAN_CHEON.finditer(text):
        n = _to_number(m["man"]) * 10_000
        if m["cheon"]:
            n += _to_number(m["cheon"]) * 1_000
        cands.append((m.start(), m.end(), _currency_str(n, "KRW")))
    for m in CURRENCY_CHEON_ONLY.finditer(text):
        cands.append((m.start(), m.end(), _currency_str(_to_number(m["n"]) * 1_000, "KRW")))
    for m in CURRENCY_EOK.finditer(text):
        cands.append((m.start(), m.end(), _currency_str(_to_number(m["n"]) * 100_000_000, "KRW")))
    for m in CURRENCY_PLAIN.finditer(text):
        cands.append((m.start(), m.end(), _currency_str(_to_number(m["n"]), CURRENCY_CODE[m["unit"]])))
    for m in CURRENCY_DOLLAR.finditer(text):
        cands.append((m.start(), m.end(), _currency_str(_to_number(m["n"]), "USD")))
    return _longest_first(cands)


# --- sys.unit.length / weight / volume --------------------------------------

LENGTH_UNIT = re.compile(
    rf"{NOT_ASCII_WORD}(?P<n>{_BNUM})\s*(?P<u>킬로미터|키로미터|센티미터|밀리미터|km|cm|mm|m|미터)(?![A-Za-z/])"
)
LENGTH_MAP = {"킬로미터": "km", "키로미터": "km", "센티미터": "cm", "밀리미터": "mm", "미터": "m",
              "km": "km", "cm": "cm", "mm": "mm", "m": "m"}
WEIGHT_UNIT = re.compile(rf"{NOT_ASCII_WORD}(?P<n>{_BNUM})\s*(?P<u>킬로그램|키로그램|그램|톤|kg|g|t)(?![A-Za-z/])")
WEIGHT_MAP = {"킬로그램": "kg", "키로그램": "kg", "그램": "g", "톤": "t", "kg": "kg", "g": "g", "t": "t"}
VOLUME_UNIT = re.compile(rf"{NOT_ASCII_WORD}(?P<n>{_BNUM})\s*(?P<u>밀리리터|리터|mL|ml|cc|L|l)(?![A-Za-z/])")
VOLUME_MAP = {"밀리리터": "ml", "리터": "l", "mL": "ml", "ml": "ml", "cc": "ml", "L": "l", "l": "l"}


def _find_unit(text: str, pattern: re.Pattern[str], unit_map: dict[str, str]) -> list[Found]:
    out = [Found(f"{_fmt_num(_to_number(m['n']))} {unit_map[m['u']]}", m.start(), m.end())
           for m in pattern.finditer(text)]
    return sorted(out, key=lambda f: f.start)


def find_length(text: str) -> list[Found]:
    return _find_unit(text, LENGTH_UNIT, LENGTH_MAP)


def find_weight(text: str) -> list[Found]:
    return _find_unit(text, WEIGHT_UNIT, WEIGHT_MAP)


def find_volume(text: str) -> list[Found]:
    return _find_unit(text, VOLUME_UNIT, VOLUME_MAP)


# --- sys.unit.temperature ----------------------------------------------------

TEMPERATURE_UNIT = re.compile(
    rf"(?P<scale>섭씨|화씨)?\s*(?P<neg>영하)?\s*(?<![0-9A-Za-z.,\-])(?P<minus>-)?(?P<n>{_BNUM})\s*(?P<u>℃|°C|°F|도)(?![A-Za-z])"
)


def find_temperature(text: str) -> list[Found]:
    out: list[Found] = []
    for m in TEMPERATURE_UNIT.finditer(text):
        n = _to_number(m["n"])
        if m["neg"] or m["minus"]:
            n = -n
        unit = "°F" if m["scale"] == "화씨" else "°C" if m["scale"] == "섭씨" else ("°F" if m["u"] == "°F" else "°C")
        out.append(Found(f"{_fmt_num(n)} {unit}", m.start(), m.end()))
    return sorted(out, key=lambda f: f.start)


# --- sys.unit.speed ----------------------------------------------------------

SPEED_UNIT = re.compile(rf"{NOT_ASCII_WORD}(?P<n>{_BNUM})\s*(?P<u>km/h|m/s)(?![A-Za-z])")
SPEED_SISOK = re.compile(rf"시속\s*(?P<n>{_BNUM})\s*(?:킬로미터|킬로|km)(?![A-Za-z])")


def find_speed(text: str) -> list[Found]:
    cands: list[tuple[int, int, int | float | str]] = []
    for m in SPEED_UNIT.finditer(text):
        cands.append((m.start(), m.end(), f"{_fmt_num(_to_number(m['n']))} {m['u']}"))
    for m in SPEED_SISOK.finditer(text):
        cands.append((m.start(), m.end(), f"{_fmt_num(_to_number(m['n']))} km/h"))
    return _longest_first(cands)


# --- normalize ---------------------------------------------------------------

_NUMBER_LIKE_FINDERS: dict[str, Callable[[str], list[Found]]] = {
    "sys.number.ordinal": find_ordinals,
    "sys.number.age": find_ages,
    "sys.number.percent": find_percents,
}
_UNIT_FINDERS: dict[str, tuple[Callable[[str], list[Found]], frozenset[str]]] = {
    "sys.unit.currency": (find_currency, frozenset({"KRW", "USD", "JPY", "EUR"})),
    "sys.unit.length": (find_length, frozenset({"mm", "cm", "m", "km"})),
    "sys.unit.weight": (find_weight, frozenset({"g", "kg", "t"})),
    "sys.unit.volume": (find_volume, frozenset({"ml", "l"})),
    "sys.unit.temperature": (find_temperature, frozenset({"°C", "°F"})),
    "sys.unit.speed": (find_speed, frozenset({"km/h", "m/s"})),
}
_PLAIN_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_UNIT_CANON = re.compile(r"(-?\d+(?:\.\d+)?)\s*(\S+)")


def _plain_number(s: str) -> int | float | None:
    if not _PLAIN_NUMBER.fullmatch(s):
        return None
    s2 = s.replace(",", "")
    return float(s2) if "." in s2 else int(s2)


def _whole_match(found: list[Found], s: str) -> Found | None:
    return next((f for f in found if f.start == 0 and f.end == len(s)), None)


def _normalize_number_like(entity: str, raw: str | int | float) -> int | float:
    if isinstance(raw, bool):
        raise ValueError(f"bool is not a valid {entity} value")
    n: int | float
    if isinstance(raw, (int, float)):
        n = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError(f"empty {entity} value")
        parsed = _plain_number(s)
        if parsed is not None:
            n = parsed
        else:
            match = _whole_match(_NUMBER_LIKE_FINDERS[entity](s), s)
            if match is None or not isinstance(match.value, (int, float)):
                raise ValueError(f"not a valid {entity} value: {raw!r}")
            n = match.value
    else:
        raise ValueError(f"invalid type for {entity}")
    if isinstance(n, float) and not math.isfinite(n):
        raise ValueError(f"{entity} value must be finite")
    if entity == "sys.number.ordinal":
        if n != int(n) or int(n) < 1:
            raise ValueError("ordinal must be an integer >= 1")
        return int(n)
    if entity == "sys.number.age":
        if n != int(n) or not (0 <= int(n) <= 150):
            raise ValueError("age must be an integer between 0 and 150")
        return int(n)
    return n  # percent: any finite number


def _normalize_date_period(raw: str | int | float) -> str:
    if not isinstance(raw, str):
        raise ValueError("date period must be a string")
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})/(\d{4})-(\d{2})-(\d{2})", raw.strip())
    if not m:
        raise ValueError(f"invalid date period: {raw!r}")
    try:
        start = dt.date(int(m[1]), int(m[2]), int(m[3]))
        end = dt.date(int(m[4]), int(m[5]), int(m[6]))
    except ValueError as e:
        raise ValueError(f"invalid date in period: {raw!r}") from e
    if start > end:
        raise ValueError(f"period end before start: {raw!r}")
    return f"{start.isoformat()}/{end.isoformat()}"


def _normalize_time_period(raw: str | int | float) -> str:
    if not isinstance(raw, str):
        raise ValueError("time period must be a string")
    m = re.fullmatch(r"(\d{1,2}):(\d{2})/(\d{1,2}):(\d{2})", raw.strip())
    if not m:
        raise ValueError(f"invalid time period: {raw!r}")
    h1, mi1, h2, mi2 = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    if not (0 <= h1 <= 23 and 0 <= mi1 <= 59 and 0 <= h2 <= 23 and 0 <= mi2 <= 59):
        raise ValueError(f"invalid time in period: {raw!r}")
    return f"{h1:02d}:{mi1:02d}/{h2:02d}:{mi2:02d}"


def _normalize_url(raw: str | int | float) -> str:
    if not isinstance(raw, str):
        raise ValueError("url must be a string")
    s = raw.strip()
    match = _whole_match(find_urls(s), s)
    if match is None:
        raise ValueError(f"invalid url: {raw!r}")
    return str(match.value)


def _normalize_unit(entity: str, raw: str | int | float) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{entity} value must be a string")
    s = raw.strip()
    if not s:
        raise ValueError(f"empty {entity} value")
    finder, units = _UNIT_FINDERS[entity]
    direct = _UNIT_CANON.fullmatch(s)
    if direct and direct[2] in units:
        return f"{_fmt_num(float(direct[1]))} {direct[2]}"
    match = _whole_match(finder(s), s)
    if match is None:
        raise ValueError(f"invalid {entity} value: {raw!r}")
    return str(match.value)


def normalize(entity: str, raw: str | int | float) -> str | int | float:
    """Validate and canonicalize a value for one of the 13 `NEW_ENTITIES`. Raises `ValueError`
    (never returns a guess) for any other entity name, or a value that entity does not accept."""
    if entity not in NEW_ENTITIES:
        raise ValueError(f"unknown entity: {entity}")
    if entity == "sys.text":
        s = str(raw).strip()
        if not s:
            raise ValueError("empty text")
        return s
    if entity in _NUMBER_LIKE_FINDERS:
        return _normalize_number_like(entity, raw)
    if entity == "sys.date.period":
        return _normalize_date_period(raw)
    if entity == "sys.time.period":
        return _normalize_time_period(raw)
    if entity == "sys.url":
        return _normalize_url(raw)
    return _normalize_unit(entity, raw)
