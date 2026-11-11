import datetime as dt
import time

import pytest

from chatcore.actions.builtin import KR_PLACES, KST, OpenMeteo, builtin_registry, korean_time, now_action
from chatcore.dialogue.values import ActionError, ActionRegistry


def fake_http(payload):
    """Stands in for the network: records requests, returns a synthetic response shaped like Open-Meteo's."""
    calls = []

    def get(url, timeout):
        calls.append((url, timeout))
        if isinstance(payload, Exception):
            raise payload
        return payload

    get.calls = calls
    return get


SYNTHETIC = {"current": {"time": "2030-01-01T09:00", "temperature_2m": 7.6, "weather_code": 61}}


def test_current_weather_maps_code_and_rounds_temperature():
    http = fake_http(SYNTHETIC)
    out = OpenMeteo(http=http, timeout=2.0).current("부산")
    assert out == {"location": "부산", "condition": "약한 비", "celsius": 8, "observed_at": "2030-01-01T09:00"}
    url, timeout = http.calls[0]
    lat, lon = KR_PLACES["부산"]
    assert f"latitude={lat}" in url and f"longitude={lon}" in url and "weather_code" in url and timeout == 2.0


def test_province_uses_its_seat_city():
    http = fake_http(SYNTHETIC)
    OpenMeteo(http=http).current("경기")
    assert f"latitude={KR_PLACES['수원'][0]}" in http.calls[0][0]


@pytest.mark.parametrize(
    "payload, message",
    [({}, "unexpected"), ({"current": {"temperature_2m": "x", "weather_code": 1}}, "unexpected"),
     ({"current": {"temperature_2m": float("nan"), "weather_code": 1}}, "no temperature")],
)
def test_bad_responses_fail_as_action_errors(payload, message):
    with pytest.raises(ActionError, match=message):
        OpenMeteo(http=fake_http(payload)).current("서울")


def test_unknown_location_and_network_errors():
    with pytest.raises(ActionError, match="no coordinates"):
        OpenMeteo(http=fake_http(SYNTHETIC)).current("아틀란티스")
    with pytest.raises(ActionError, match="down"):
        OpenMeteo(http=fake_http(ActionError("down"))).current("서울")


def test_unknown_weather_code_is_reported_plainly():
    out = OpenMeteo(http=fake_http({"current": {"temperature_2m": 1, "weather_code": 42}})).current("서울")
    assert out["condition"] == "알 수 없는 날씨"


@pytest.mark.parametrize("h, m, text", [(0, 0, "오전 12시"), (9, 0, "오전 9시"), (12, 30, "오후 12시 30분"), (15, 5, "오후 3시 5분"),
                                       (23, 59, "오후 11시 59분")])
def test_korean_time(h, m, text):
    assert korean_time(dt.datetime(2030, 1, 1, h, m)) == text


def test_now_uses_korea_time():
    utc_noon = dt.datetime(2030, 1, 1, 3, 0, tzinfo=dt.UTC)  # 12:00 in Seoul
    assert now_action(lambda: utc_noon)({}) == {"time": "오후 12시", "iso": "2030-01-01T12:00+09:00"}


def test_builtin_registry_validates_args_and_runs():
    reg = builtin_registry(OpenMeteo(http=fake_http(SYNTHETIC)), clock=lambda: dt.datetime(2030, 1, 1, tzinfo=KST))
    assert reg.names() == {"get_weather", "get_temperature", "now", "knowledge"}
    assert reg.run("knowledge", {}) == {"found": "no", "answer": "", "source": "", "title": ""}  # no knowledge base
    assert reg.run("get_temperature", {"location": "서울"})["celsius"] == 8
    with pytest.raises(ActionError, match="bad arguments"):
        reg.run("get_weather", {})


def test_registry_timeout_is_distinct_from_an_actions_own_timeout_error():
    reg = ActionRegistry(timeout=0.05)
    reg.register("slow", lambda a: time.sleep(0.5) or {})
    reg.register("raises", lambda a: (_ for _ in ()).throw(TimeoutError("own")))
    with pytest.raises(ActionError, match="timed out after 0.05s"):
        reg.run("slow", {})
    with pytest.raises(ActionError, match="raised TimeoutError: own"):
        reg.run("raises", {})
    with pytest.raises(ValueError):
        ActionRegistry(timeout=0)


def test_district_uses_its_city():
    http = fake_http(SYNTHETIC)
    OpenMeteo(http=http).current("서울 강남구")
    assert f"latitude={KR_PLACES['서울'][0]}" in http.calls[0][0]
    with pytest.raises(ActionError):
        OpenMeteo(http=http).current("고양 덕양구")


def test_pool_is_created_once_under_concurrency():
    import threading

    reg = ActionRegistry(timeout=1.0)
    reg.register("ok", lambda a: {"v": 1})
    threads = [threading.Thread(target=reg.run, args=("ok", {})) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert reg._pool is not None
