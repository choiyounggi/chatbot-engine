import pytest

from chatcore.bot.josa import attach


@pytest.mark.parametrize(
    "word, particle, expected",
    [("부산", "은", "부산은"), ("대구", "은", "대구는"), ("대구", "는", "대구는"), ("제주", "이", "제주가"), ("서울", "가", "서울이"),
     ("맑음", "이에요", "맑음이에요"), ("비", "이에요", "비예요"), ("오후 3시", "이에요", "오후 3시예요"),
     ("오후 3시 5분", "이에요", "오후 3시 5분이에요"), ("서울", "으로", "서울로"), ("부산", "로", "부산으로"), ("대구", "으로", "대구로"),
     ("3", "이", "3이"), ("2", "이", "2가"), ("10", "을", "10을"), ("12°C", "이에요", "12°C예요"), ("커피", "와", "커피와"),
     ("빵", "와", "빵과")],
)
def test_attach(word, particle, expected):
    assert attach(word, particle) == expected


def test_unknown_final_sound_gets_both_forms():
    assert attach("PC", "은") == "PC은(는)" and attach("", "이") == "이(가)"
    assert attach("Kim", "은") == "Kim은(는)"  # ends in m but is not a unit after a number
    assert attach("5km", "이에요") == "5km예요" and attach("30%", "이") == "30%가"


def test_unknown_particle_raises():
    with pytest.raises(ValueError, match="unknown particle"):
        attach("부산", "께서")
