"""Korean particles that depend on the preceding word's final consonant (받침).

`attach("부산", "은")` -> "부산은", `attach("대구", "은")` -> "대구는". Either form
of a pair may be written in a template ({{ slots.location|은 }} or |는).
"""

from __future__ import annotations

import re

# (after a final consonant, after a vowel)
PAIRS = {
    "은": ("은", "는"), "는": ("은", "는"),
    "이": ("이", "가"), "가": ("이", "가"),
    "을": ("을", "를"), "를": ("을", "를"),
    "과": ("과", "와"), "와": ("과", "와"),
    "이에요": ("이에요", "예요"), "예요": ("이에요", "예요"),
    "이야": ("이야", "야"), "야": ("이야", "야"),
    "이랑": ("이랑", "랑"), "랑": ("이랑", "랑"),
    "으로": ("으로", "로"), "로": ("으로", "로"),
}
PARTICLES = frozenset(PAIRS)
RIEUL = 8  # jongseong index of ㄹ
# Final consonant of the Korean reading of each digit (영 일 이 삼 사 오 육 칠 팔 구): 0 = none.
DIGIT_FINAL = {"0": 21, "1": RIEUL, "2": 0, "3": 16, "4": 0, "5": 0, "6": 1, "7": RIEUL, "8": RIEUL, "9": 0}
# Units read aloud after a number ("12°C" -> 도, "3%" -> 퍼센트, "5km" -> 킬로미터)
UNIT_AFTER_NUMBER = re.compile(r"\d\s*(?:°C|°|%|km|m)$")


def _final(word: str) -> int | None:
    """Jongseong index of the word's last sound (0 = vowel ending), None if unknown."""
    word = word.rstrip()
    if not word:
        return None
    if UNIT_AFTER_NUMBER.search(word):  # only after a digit: "Kim" is not a unit
        return 0
    last = word[-1]
    if "가" <= last <= "힣":
        return (ord(last) - 0xAC00) % 28
    if last in DIGIT_FINAL:
        return DIGIT_FINAL[last]
    return None  # Latin letters, symbols: how it is read aloud is unknown


def attach(word: str, particle: str) -> str:
    if particle not in PAIRS:
        raise ValueError(f"unknown particle {particle!r}; use one of {sorted(PARTICLES)}")
    with_final, without = PAIRS[particle]
    final = _final(word)
    if final is None:
        return f"{word}{with_final}({without})" if with_final != without else word + particle
    if with_final == "으로":  # ㄹ-final words take 로: 서울로
        return word + ("로" if final in (0, RIEUL) else "으로")
    return word + (with_final if final else without)
