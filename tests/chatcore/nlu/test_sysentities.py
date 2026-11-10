import datetime as dt

import pytest

from chatcore.nlu.sysentities import (
    find_ages,
    find_currency,
    find_date_periods,
    find_dates,
    find_length,
    find_numbers,
    find_ordinals,
    find_percents,
    find_speed,
    find_temperature,
    find_time_periods,
    find_times,
    find_urls,
    find_volume,
    find_weight,
    normalize,
)

TODAY = dt.date(2026, 11, 10)  # a Tuesday


@pytest.mark.parametrize(
    "text, values",
    [("3개", [3]), ("1,200원", [1200]), ("3.5도", [3.5]), ("영하 5도", [-5]), ("-2", [-2]), ("사과 세 개", [3]),
     ("열두 명", [12]), ("두 번", [2]), ("숫자 없음", []), ("", []), ("-2도", [-2]), ("세 개를 줘", [3]),
     ("세 명이서", [3]), ("두 번만", [2]), ("사과3개", [3]),
     # never a wrong value: unrecognized instead
     ("열세 개", []), ("스물다섯 명", []), ("서른두 명", []), ("정한 시간", []), ("건강한 개", []), ("네 번호는 1234", [1234]),
     ("2026-11-10", [2026]), ("3-4명", [3]), ("a-3", [])],
)
def test_numbers(text, values):
    assert [f.value for f in find_numbers(text)] == values


@pytest.mark.parametrize(
    "text, iso",
    [("2026-12-25", "2026-12-25"), ("2027년 1월 2일", "2027-01-02"), ("11월 9일", "2026-11-09"),
     ("1월 3일", "2027-01-03"), ("3일 후", "2026-11-13"), ("오늘", "2026-11-10"), ("내일", "2026-11-11"),
     ("모레", "2026-11-12"), ("어제", "2026-11-09"), ("그저께", "2026-11-08"), ("금요일", "2026-11-13"),
     ("화요일", "2026-11-10"), ("이번 주 월요일", "2026-11-09"), ("다음 주 금요일", "2026-11-20"),
     ("담주 월요일", "2026-11-16"), ("지난주 금요일", "2026-11-06"), ("다음 금요일", "2026-11-13"),
     ("다음 화요일", "2026-11-17"), ("지난 금요일", "2026-11-06"), ("지난 화요일", "2026-11-03"), ("이번 금요일", "2026-11-13")],
)
def test_dates(text, iso):
    assert [f.value for f in find_dates(text, TODAY)] == [iso]


@pytest.mark.parametrize("text", ["2월 30일", "2026-13-01", "날짜 없음", ""])
def test_invalid_or_missing_dates(text):
    assert find_dates(text, TODAY) == []


@pytest.mark.parametrize(
    "text, hhmm",
    [("15:30", "15:30"), ("3시", "03:00"), ("오후 3시 반", "15:30"), ("저녁 7시 10분", "19:10"), ("오전 12시", "00:00"),
     ("낮 2시", "14:00"), ("밤 11시", "23:00"), ("정오", "12:00"), ("자정", "00:00"), ("밤 12시", "00:00"),
     ("밤 1시", "01:00"), ("밤 9시", "21:00"), ("오후 3:30", "15:30"), ("오전 12:10", "00:10"), ("새벽 2시", "02:00")],
)
def test_times(text, hhmm):
    assert [f.value for f in find_times(text)] == [hhmm]


def test_time_span_excludes_leading_space_and_invalid_hours():
    found = find_times("금요일 2시")
    assert found[0].start == 4 and "금요일 2시"[found[0].start:found[0].end] == "2시"
    assert find_times("25시") == [] and find_times("3시 75분") == []


@pytest.mark.parametrize("text", ["3시간 걸려요", "1시간 30분"])
def test_durations_are_not_times(text):
    assert find_times(text) == []


def test_weekday_span_has_no_leading_space():
    found = find_dates("오는 금요일", TODAY)
    assert "오는 금요일"[found[0].start:found[0].end] == "금요일"


@pytest.mark.parametrize(
    "text, values",
    [("첫 번째", [1]), ("첫번째", [1]), ("첫째", [1]), ("두 번째", [2]), ("세 번째", [3]), ("둘째", [2]), ("셋째", [3]),
     ("열째", [10]), ("3번째", [3]), ("10 번째", [10]), ("열두 번째", [12]), ("스물세 번째", [23]), ("스무 번째", [20]),
     ("열 번째", [10]), ("열둘째", []),  # a compound "째" form is left alone rather than read as 2
     # must not match
     ("3번 출구", []), ("두 번", []), ("", [])],
)
def test_ordinals(text, values):
    assert [f.value for f in find_ordinals(text)] == values


@pytest.mark.parametrize(
    "text, values",
    [("25살", [25]), ("25 세", [25]), ("만 25세", [25]), ("스무 살", [20]), ("스물다섯 살", [25]), ("서른두살", [32]),
     ("다섯 살", [5]), ("열 살", [10]), ("0살", [0]), ("150살", [150]),
     # must not match / out of range
     ("세 개", []), ("25세기", []), ("세계", []), ("151살", []), ("3세트", []), ("2세대", []), ("", []),
     ("25세예요", [25]), ("3살이야", [3])],
)
def test_ages(text, values):
    assert [f.value for f in find_ages(text)] == values


@pytest.mark.parametrize(
    "text, values",
    [("20%", [20]), ("3.5%", [3.5]), ("50퍼센트", [50]), ("10프로", [10]),
     # must not match
     ("프로그램", []), ("3프로젝트", []), ("", []), ("30프로야", [30])],
)
def test_percents(text, values):
    assert [f.value for f in find_percents(text)] == values


@pytest.mark.parametrize(
    "text, value",
    [("11월 5일부터 10일까지", "2026-11-05/2026-11-10"), ("11월 5일부터 12월 10일까지", "2026-11-05/2026-12-10"),
     ("이번 주", "2026-11-09/2026-11-15"), ("다음 주", "2026-11-16/2026-11-22"), ("담주", "2026-11-16/2026-11-22"),
     ("지난주", "2026-11-02/2026-11-08"), ("저번 주", "2026-11-02/2026-11-08"), ("이번 달", "2026-11-01/2026-11-30"),
     ("다음 달", "2026-12-01/2026-12-31"), ("지난달", "2026-10-01/2026-10-31"), ("주말", "2026-11-14/2026-11-15"),
     ("이번 주말", "2026-11-14/2026-11-15"), ("다음 주말", "2026-11-21/2026-11-22"), ("올해", "2026-01-01/2026-12-31"),
     ("내년", "2027-01-01/2027-12-31"), ("작년", "2025-01-01/2025-12-31"),
     ("11월 5일부터 5일까지", "2026-11-05/2026-11-05")],  # boundary: end == start
)
def test_date_periods(text, value):
    assert [f.value for f in find_date_periods(text, TODAY)] == [value]


@pytest.mark.parametrize(
    "text",
    ["다음 주 월요일", "다음주 금요일",  # single dates, not periods
     "11월 10일부터 11월 5일까지",  # boundary: end before start is not recognized
     ""],
)
def test_date_periods_not_recognized(text):
    assert find_date_periods(text, TODAY) == []


@pytest.mark.parametrize(
    "text, value",
    [("오후 2시부터 4시까지", "14:00/16:00"), ("밤 10시부터 새벽 2시까지", "22:00/02:00"),
     ("2시부터 2시까지", "02:00/02:00"),  # boundary: end == start
     ("오후 11시부터 2시까지", "23:00/02:00"), ("밤 10시부터 2시까지", "22:00/02:00")],  # +12h would end before it starts
)
def test_time_periods(text, value):
    assert [f.value for f in find_time_periods(text)] == [value]


@pytest.mark.parametrize("text", ["3시", ""])  # a single time is not a period
def test_time_periods_not_recognized(text):
    assert find_time_periods(text) == []


@pytest.mark.parametrize(
    "text, value",
    [("http://a.com", "http://a.com"), ("https://a.com/path?a=1.", "https://a.com/path?a=1"),
     ("www.naver.com", "https://www.naver.com"), ("www.naver.com에서", "https://www.naver.com"),
     ("https://a.com에 접속", "https://a.com")],  # a URL ends where Korean starts
)
def test_urls(text, value):
    assert [f.value for f in find_urls(text)] == [value]


@pytest.mark.parametrize("text", ["example.com", ""])
def test_urls_not_recognized(text):
    assert find_urls(text) == []


@pytest.mark.parametrize(
    "text, value",
    [("3만 원", "30000 KRW"), ("5천원", "5000 KRW"), ("1만5천원", "15000 KRW"), ("2억 원", "200000000 KRW"),
     ("3만 원이요", "30000 KRW"),
     ("100달러", "100 USD"), ("$50", "50 USD"), ("3.5달러", "3.5 USD")],  # boundary: a decimal amount
)
def test_currency(text, value):
    assert [f.value for f in find_currency(text)] == [value]


@pytest.mark.parametrize("text", ["원래", "만원", "", "5억 명이", "3원칙", "9" * 310 + "원"])
def test_currency_not_recognized(text):
    assert find_currency(text) == []


@pytest.mark.parametrize(
    "text, value",
    [("5km", "5 km"), ("5mm", "5 mm"), ("3킬로미터", "3 km"), ("3키로미터", "3 km"), ("1.5km", "1.5 km")],
)
def test_length(text, value):
    assert [f.value for f in find_length(text)] == [value]


@pytest.mark.parametrize("text", ["5min", "", "100km/h", "5m/s", "9" * 310 + "km"])
def test_length_not_recognized(text):
    assert find_length(text) == []


@pytest.mark.parametrize(
    "text, value", [("1.5kg", "1.5 kg"), ("500g", "500 g"), ("1톤", "1 t"), ("1키로그램", "1 kg")]
)
def test_weight(text, value):
    assert [f.value for f in find_weight(text)] == [value]


@pytest.mark.parametrize("text", [""])
def test_weight_not_recognized(text):
    assert find_weight(text) == []


@pytest.mark.parametrize(
    "text, value", [("500ml", "500 ml"), ("2L", "2 l"), ("1cc", "1 ml"), ("1리터", "1 l")]
)
def test_volume(text, value):
    assert [f.value for f in find_volume(text)] == [value]


@pytest.mark.parametrize("text", [""])
def test_volume_not_recognized(text):
    assert find_volume(text) == []


@pytest.mark.parametrize(
    "text, value",
    [("25도", "25 °C"), ("섭씨 36.5도", "36.5 °C"), ("영하 5도", "-5 °C"), ("화씨 100도", "100 °F"), ("25℃", "25 °C"),
     ("-5도", "-5 °C"), ("1,000도", "1000 °C")],
)
def test_temperature(text, value):
    assert [f.value for f in find_temperature(text)] == [value]


@pytest.mark.parametrize("text", ["서울도", ""])
def test_temperature_not_recognized(text):
    assert find_temperature(text) == []


@pytest.mark.parametrize("text, value", [("100km/h", "100 km/h"), ("5m/s", "5 m/s"), ("시속 60킬로", "60 km/h")])
def test_speed(text, value):
    assert [f.value for f in find_speed(text)] == [value]


@pytest.mark.parametrize("text", [""])
def test_speed_not_recognized(text):
    assert find_speed(text) == []


@pytest.mark.parametrize(
    "entity, raw, expected",
    [("sys.text", "  hi  ", "hi"), ("sys.number.ordinal", 3, 3), ("sys.number.ordinal", "세 번째", 3),
     ("sys.number.age", "25살", 25), ("sys.number.age", 0, 0), ("sys.number.percent", "20%", 20),
     ("sys.number.percent", 3.5, 3.5),
     ("sys.date.period", "2026-11-05/2026-11-10", "2026-11-05/2026-11-10"),
     ("sys.time.period", "9:05/14:00", "09:05/14:00"),
     ("sys.url", "www.naver.com", "https://www.naver.com"),
     ("sys.unit.currency", "3만원", "30000 KRW"), ("sys.unit.length", "3킬로미터", "3 km"),
     ("sys.unit.weight", "1.5kg", "1.5 kg"), ("sys.unit.volume", "1리터", "1 l"),
     ("sys.unit.temperature", "섭씨 36.5도", "36.5 °C"), ("sys.unit.speed", "시속 60킬로", "60 km/h")],
)
def test_normalize_valid(entity, raw, expected):
    assert normalize(entity, raw) == expected


@pytest.mark.parametrize(
    "entity, raw",
    [("sys.text", "   "), ("sys.number.ordinal", 0), ("sys.number.ordinal", 1.5), ("sys.number.ordinal", True),
     ("sys.number.age", 200), ("sys.number.age", -1), ("sys.number.percent", "not a percent"),
     ("sys.date.period", "2026-11-10/2026-11-05"), ("sys.date.period", "not a period"),
     ("sys.time.period", "25:00/14:00"), ("sys.url", "not a url"), ("sys.unit.length", "3 kg"),
     ("sys.unit.weight", "3 m"), ("sys.unit.volume", "3 g"), ("sys.unit.temperature", "hot"),
     ("sys.unit.speed", "3 kg"), ("sys.unit.currency", "만원"), ("unknown.entity", 5), ("sys.number.ordinal", "")],
)
def test_normalize_invalid(entity, raw):
    with pytest.raises(ValueError):
        normalize(entity, raw)
