import datetime as dt

import pytest

from chatcore.nlu.sysentities import find_dates, find_numbers, find_times

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
