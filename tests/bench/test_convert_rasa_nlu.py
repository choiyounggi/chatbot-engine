from pathlib import Path

import pytest

from bench.convert_rasa_nlu import (
    ConversionError,
    build_document,
    convert,
    group_synonyms,
    load_additions,
    merge_additions,
    read_legacy_keywords,
    read_rasa_examples,
    to_template,
)


def test_annotation_becomes_placeholder_and_value():
    assert to_template("오늘 [가상시](location) 날씨") == ("오늘 {location} 날씨", ["가상시"])


def test_deictic_location_stays_plain_text():
    assert to_template("[여기](location) 비 오나요?") == ("여기 비 오나요?", [])


def test_unsupported_entity_type_raises():
    with pytest.raises(ConversionError, match="unsupported entity"):
        to_template("[내일](date) 날씨")


@pytest.mark.parametrize("text", ["", "   ", "[ ](location)"])
def test_empty_example_or_value_raises(text):
    with pytest.raises(ConversionError):
        to_template(text)


def test_unknown_location_is_routed_by_temperature_words():
    conv = convert(
        [("unknown_location", "[저기](location) 기온이 어때"), ("unknown_location", "[거기](location) 비 와?")],
        legacy_keywords=["가상"],
    )
    assert conv.intents == {"temperature": ["저기 기온이 어때"], "weather": ["거기 비 와?"]}
    assert conv.location_values == {"가상"}  # deictics are not values; legacy keywords are merged


def test_duplicate_templates_collapse_within_intent():
    conv = convert(
        [("weather", "[가상시](location) 날씨"), ("weather", "[다른시](location) 날씨")],
        legacy_keywords=[],
    )
    assert conv.intents == {"weather": ["{location} 날씨"]}
    assert conv.location_values == {"가상시", "다른시"}


def test_known_cross_intent_conflict_is_resolved_and_noted():
    conv = convert([("greeting", "안녕"), ("bye", "안녕")], legacy_keywords=[])
    assert conv.intents == {"greeting": ["안녕"]}
    assert any("drop conflicting" in n for n in conv.notes)


def test_unknown_cross_intent_conflict_raises():
    with pytest.raises(ConversionError, match="both"):
        convert([("thanks", "좋아요"), ("compliment", "좋아요")], legacy_keywords=[])


def test_templates_colliding_across_intents_raise():
    with pytest.raises(ConversionError, match="several intents"):
        convert([("weather", "[가상시](location) 어때"), ("temperature", "[다른시](location) 어때")], legacy_keywords=[])


def test_known_template_collision_moves_to_owner():
    conv = convert(
        [("weather", "[가상](location)는 지금 몇 도에요?"), ("weather", "비 와?"), ("temperature", "기온 알려줘")],
        legacy_keywords=[],
    )
    assert conv.intents == {"weather": ["비 와?"], "temperature": ["기온 알려줘", "{location}는 지금 몇 도에요?"]}


def test_read_rasa_examples_rejects_non_list_lines(tmp_path: Path):
    f = tmp_path / "nlu.yml"
    f.write_text('nlu:\n- intent: greeting\n  examples: |\n    - 안녕\n    "반가워"\n', encoding="utf-8")
    with pytest.raises(ConversionError, match="not '- ' examples"):
        read_rasa_examples(f)


def test_synonym_grouping():
    groups = group_synonyms({"가상", "가상시", "가상광역시", "충청북도", "충북", "강원", "강원도", "전라도", "청도", "가상 북구"})
    assert groups["가상"] == ["가상광역시", "가상시"]
    assert groups["충북"] == ["충청북도"]
    assert groups["강원"] == ["강원도"]
    assert groups["전라도"] == []  # "전라" is not a known value
    assert groups["청도"] == []  # one-letter base is not stripped
    assert groups["가상 북구"] == []


def test_build_document_rejects_oos_overlap():
    conv = convert([("weather", "날씨 어때")], legacy_keywords=["가상"])
    with pytest.raises(ConversionError, match="overlap"):
        build_document(conv, ["날씨 어때"])


def test_read_rasa_examples_rejects_empty_intent(tmp_path: Path):
    f = tmp_path / "nlu.yml"
    f.write_text('nlu:\n- intent: greeting\n  examples: ""\n', encoding="utf-8")
    with pytest.raises(ConversionError, match="no examples"):
        read_rasa_examples(f)


def test_read_legacy_keywords_dedupes_and_requires_array(tmp_path: Path):
    java = tmp_path / "S.java"
    java.write_text('String[] LOCATION_KEYWORDS = {\n "가", "나", "가"\n};', encoding="utf-8")
    assert read_legacy_keywords(java) == ["가", "나"]
    java.write_text("class S {}", encoding="utf-8")
    with pytest.raises(ConversionError):
        read_legacy_keywords(java)


def test_drop_table_removes_example_and_notes_it():
    conv = convert([("greeting", "오늘 날씨 좋네요"), ("greeting", "안녕하세요")], legacy_keywords=[])
    assert conv.intents == {"greeting": ["안녕하세요"]}
    assert any("drop from greeting" in n for n in conv.notes)


def test_merge_additions_appends_new_examples():
    conv = convert([("help", "도움말")], legacy_keywords=[])
    merge_additions(conv, {"help": ["헬프"]})
    assert conv.intents["help"] == ["도움말", "헬프"]


@pytest.mark.parametrize(
    "additions, match",
    [({"nope": ["예문"]}, "unknown intent"), ({"help": ["도움말"]}, "already exists"), ({"help": ["새것", "새것"]}, "already exists")],
)
def test_merge_additions_rejects_unknown_intent_and_duplicates(additions, match):
    conv = convert([("help", "도움말")], legacy_keywords=[])
    with pytest.raises(ConversionError, match=match):
        merge_additions(conv, additions)


def test_load_additions_rejects_empty_lists(tmp_path: Path):
    f = tmp_path / "add.yaml"
    f.write_text("intents:\n  help: []\n", encoding="utf-8")
    with pytest.raises(ConversionError):
        load_additions(f)


def test_build_document_rejects_heldout_sentences_in_training():
    conv = convert([("time", "지금 몇 시야")], legacy_keywords=["가상"])
    with pytest.raises(ConversionError, match="held-out"):
        build_document(conv, ["주식 추천해줘"], oos_eval=["지금 몇 시야"])
