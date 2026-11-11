"""Built-in actions: current weather (Open-Meteo, no API key) and the current time.

Locations are looked up in a fixed coordinate table instead of the Open-Meteo
geocoder: searching Korean names there returns nothing for "서울" and a
mountain in Chungbuk for "부산" (checked 2026-11-10). The table holds the
largest populated place for each English name in Open-Meteo/GeoNames
geocoding (countryCode=KR), fetched 2026-11-10; provinces map to the city
hosting the provincial government. Other locations are reported as
unsupported instead of guessed.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from chatcore.dialogue.values import ActionError, ActionRegistry

KST = ZoneInfo("Asia/Seoul")

KR_PLACES: dict[str, tuple[float, float]] = {
    "서울": (37.5660, 126.9784), "부산": (35.1017, 129.0300), "대구": (35.8710, 128.6012),
    "인천": (37.4565, 126.7052), "광주": (35.1547, 126.9156), "대전": (36.3502, 127.3848),
    "울산": (35.5392, 129.3124), "세종": (36.5924, 127.2922), "제주": (33.5097, 126.5219),
    "수원": (37.2782, 127.0316), "춘천": (37.8830, 127.7289), "강릉": (37.7527, 128.8724),
    "청주": (36.6372, 127.4897), "천안": (36.8065, 127.1522), "전주": (35.8248, 127.1480),
    "포항": (36.0292, 129.3648), "창원": (35.2281, 128.6811), "원주": (37.3514, 127.9453),
    "목포": (34.8128, 126.3918), "여수": (34.7606, 127.6621), "경주": (35.8428, 129.2117),
    "안동": (36.5680, 128.7297), "진주": (35.1928, 128.0847), "속초": (38.2070, 128.5918),
    "홍성": (36.6016, 126.6614), "무안": (34.9901, 126.4790),
}
PROVINCE_SEAT = {"경기": "수원", "강원": "춘천", "충북": "청주", "충남": "홍성", "전북": "전주", "전남": "무안",
                 "경북": "안동", "경남": "창원"}

# WMO weather interpretation codes (Open-Meteo docs) -> everyday Korean
WMO = {0: "맑음", 1: "대체로 맑음", 2: "구름 조금", 3: "흐림", 45: "안개", 48: "안개",
       51: "약한 이슬비", 53: "이슬비", 55: "강한 이슬비", 56: "어는 이슬비", 57: "어는 이슬비",
       61: "약한 비", 63: "비", 65: "강한 비", 66: "어는 비", 67: "어는 비",
       71: "약한 눈", 73: "눈", 75: "많은 눈", 77: "싸락눈", 80: "약한 소나기", 81: "소나기", 82: "강한 소나기",
       85: "눈 소나기", 86: "강한 눈 소나기", 95: "뇌우", 96: "우박을 동반한 뇌우", 99: "우박을 동반한 뇌우"}

HttpGet = Callable[[str, float], Any]  # (url, timeout seconds) -> parsed JSON


def http_get_json(url: str, timeout: float) -> Any:  # pragma: no cover - network; tests inject a fake
    request = urllib.request.Request(url, headers={"User-Agent": "chatcore/0.0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ActionError(f"weather service unavailable: {exc}") from exc


class LocationArgs(BaseModel):
    location: str


class OpenMeteo:
    URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, places: Mapping[str, tuple[float, float]] | None = None, http: HttpGet = http_get_json,
                 timeout: float = 4.0) -> None:
        self.places = dict(places if places is not None else KR_PLACES)
        self.http = http
        self.timeout = timeout

    def coordinates(self, location: str) -> tuple[float, float]:
        """Coordinates of a city, a province (its seat) or a district ("서울 강남구" -> 서울)."""
        for name in (location, location.split()[0] if location.split() else location):
            place = PROVINCE_SEAT.get(name, name)
            if place in self.places:
                return self.places[place]
        raise ActionError(f"no coordinates for {location!r}")

    def current(self, location: str) -> dict[str, Any]:
        lat, lon = self.coordinates(location)
        query = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code",
                                        "timezone": "Asia/Seoul"})
        data = self.http(f"{self.URL}?{query}", self.timeout)
        try:
            current = data["current"]
            temp, code = float(current["temperature_2m"]), int(current["weather_code"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionError(f"unexpected weather response: {exc}") from exc
        if not math.isfinite(temp):
            raise ActionError("weather response has no temperature")
        return {"location": location, "condition": WMO.get(code, "알 수 없는 날씨"), "celsius": round(temp),
                "observed_at": str(current.get("time", ""))}

    def weather(self, args: dict[str, str]) -> dict[str, Any]:
        return self.current(args["location"])

    def temperature(self, args: dict[str, str]) -> dict[str, Any]:
        return self.current(args["location"])


def korean_time(moment: dt.datetime) -> str:
    """'오전 12시' (midnight), '오전 9시', '오후 3시 5분' style."""
    hour, minute = moment.hour, moment.minute
    part = "오전" if hour < 12 else "오후"
    h12 = hour % 12 or 12
    return f"{part} {h12}시" + (f" {minute}분" if minute else "")


def now_action(clock: Callable[[], dt.datetime]) -> Callable[[dict[str, str]], dict[str, Any]]:
    def now(args: dict[str, str]) -> dict[str, Any]:
        moment = clock().astimezone(KST)
        return {"time": korean_time(moment), "iso": moment.isoformat(timespec="minutes")}

    return now


NOT_FOUND = {"found": "no", "answer": "", "source": "", "title": ""}  # the `knowledge` action's result shape


def builtin_registry(weather: OpenMeteo | None = None, clock: Callable[[], dt.datetime] | None = None,
                     timeout: float = 5.0) -> ActionRegistry:
    """get_weather, get_temperature (args: location), now, and knowledge (a bot with a knowledge base
    gets its own: chatcore.knowledge.knowledge_action; without one nothing is found)."""
    weather = weather or OpenMeteo()
    registry = ActionRegistry(timeout=timeout)
    registry.register("get_weather", weather.weather, LocationArgs)
    registry.register("get_temperature", weather.temperature, LocationArgs)
    registry.register("now", now_action(clock or (lambda: dt.datetime.now(KST))))
    registry.register("knowledge", lambda args: dict(NOT_FOUND))
    return registry
