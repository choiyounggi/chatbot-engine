from pathlib import Path

import pytest
import yaml

from chatcore.bot.loader import BotError, load_bot, validate
from chatcore.bot.schema import BotDefinition, CollectStep, DictionaryEntity

REQUIRED = {k: {"text": k} for k in ("fallback", "cancelled", "restarted", "error", "clarify", "input_invalid")}


def bot(**over) -> BotDefinition:
    data = {
        "id": "t", "version": "v",
        "intents": {"a": {"examples": ["1", "2", "3", "4", "5"]}, "b": {"examples": ["6", "7", "8", "9", "10"]}},
        "entities": {"city": {"kind": "dictionary", "values": {"서울": ["서울시"], "부산": []}}},
        "scenarios": {"s": {"trigger": {"intent": "a"}, "slots": {"c": {"entity": "city", "ask": "ask"}},
                            "steps": [{"collect": "c"}, {"respond": "done"}]}},
        "responses": {**REQUIRED, "ask": {"text": "어디?"}, "done": {"text": "끝"}},
    }
    for key, value in over.items():
        data[key] = value
    return BotDefinition.model_validate(data)


def write_bot(tmp_path: Path, bot_yaml: dict, nlu_yaml: dict | None = None) -> Path:
    (tmp_path / "bot.yaml").write_text(yaml.safe_dump(bot_yaml, allow_unicode=True), encoding="utf-8")
    if nlu_yaml is not None:
        (tmp_path / "nlu.yaml").write_text(yaml.safe_dump(nlu_yaml, allow_unicode=True), encoding="utf-8")
    return tmp_path


def test_sample_bot_loads_with_its_actions(sample_bot):
    assert sample_bot.id == "sample" and len(sample_bot.version) == 12
    assert isinstance(sample_bot.scenarios["check_weather"].steps[0], CollectStep)
    assert validate(sample_bot).errors == []


def test_valid_bot_has_no_errors():
    assert validate(bot()).errors == []


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"responses": {"ask": {"text": "x"}, "done": {"text": "y"}}}, "missing required responses"),
        ({"scenarios": {"s": {"trigger": {"intent": "zzz"}, "steps": [{"respond": "done"}]}}}, "trigger intent 'zzz'"),
        ({"scenarios": {"s": {"trigger": {"intent": "a"}, "steps": [{"respond": "done"}]},
                        "s2": {"trigger": {"intent": "a"}, "steps": [{"respond": "done"}]}}}, "already triggers"),
        ({"scenarios": {"s": {"slots": {"c": {"entity": "nope", "ask": "ask"}}, "steps": [{"collect": "c"}]}}}, "unknown entity"),
        ({"scenarios": {"s": {"slots": {"c": {"entity": "city", "ask": "nope"}}, "steps": [{"collect": "c"}]}}}, "unknown response 'nope'"),
        ({"scenarios": {"s": {"steps": [{"collect": "c"}]}}}, "undeclared slot"),
        ({"scenarios": {"s": {"steps": [{"respond": "nope"}]}}}, "unknown response 'nope'"),
        ({"scenarios": {"s": {"steps": [{"call": "nope"}]}}}, "calls unknown scenario"),
        ({"scenarios": {"s": {"steps": [{"branch": {"on": "slots.x", "cases": {"1": "nope"}}}]}}}, "branches to unknown"),
        ({"scenarios": {"s": {"on_error": "nope", "steps": [{"respond": "done"}]}}}, "on_error response"),
        ({"scenarios": {"s": {"steps": [{"call": "t"}]}, "t": {"steps": [{"branch": {"on": "x", "cases": {"1": "s"}}}]}}}, "cycle"),
        ({"entities": {"c1": {"kind": "dictionary", "values": {"서울": []}},
                       "c2": {"kind": "dictionary", "values": {"수도": ["서울"]}}},
          "scenarios": {"s": {"steps": [{"respond": "done"}]}}}, "belongs to both"),
        ({"intents": {"a": {"examples": ["같은 문장"]}, "b": {"examples": ["같은 문장"]}},
          "scenarios": {"s": {"steps": [{"respond": "done"}]}}}, "is in both intents"),
    ],
)
def test_validation_errors(over, needle):
    errors = validate(bot(**over)).errors
    assert any(needle in e for e in errors), errors


def test_unknown_action_only_checked_when_actions_given():
    b = bot(scenarios={"s": {"steps": [{"action": "fetch"}]}})
    assert validate(b).errors == []
    assert any("unknown action 'fetch'" in e for e in validate(b, actions={"other"}).errors)


def test_self_call_is_a_cycle():
    assert any("s -> s" in e for e in validate(bot(scenarios={"s": {"steps": [{"call": "s"}]}})).errors)


def test_few_examples_is_a_warning_not_error():
    report = validate(bot(intents={"a": {"examples": ["하나"]}}))
    assert report.errors == [] and any("only 1 examples" in w for w in report.warnings)


@pytest.mark.parametrize(
    "steps",
    [[{"collect": "c", "respond": "done"}], [{"unknown": 1}], ["respond"], []],
)
def test_malformed_steps_are_rejected(steps):
    with pytest.raises(ValueError):
        bot(scenarios={"s": {"steps": steps}})


@pytest.mark.parametrize("resp", [{"text": "a", "variants": ["b"]}, {}, {"variants": []}, {"text": "  "}, {"variants": ["ok", " "]}])
def test_malformed_responses_are_rejected(resp):
    with pytest.raises(ValueError):
        bot(responses={**REQUIRED, "ask": resp, "done": {"text": "끝"}})


def test_bad_ids_and_regex_are_rejected():
    with pytest.raises(ValueError):
        bot(scenarios={"Bad-Id": {"steps": [{"respond": "done"}]}})
    with pytest.raises(ValueError, match="invalid regex"):
        bot(entities={"p": {"kind": "regex", "pattern": "("}})


def test_dictionary_canonical_lookup():
    ent = DictionaryEntity(kind="dictionary", values={"서울": ["서울시"]})
    assert ent.canonical("서울시") == "서울" and ent.canonical("서울") == "서울" and ent.canonical("부산") is None


def test_load_bot_merges_files_and_hashes_content(tmp_path):
    b = bot()
    raw = b.model_dump(exclude={"version", "intents", "entities"})
    nlu = b.model_dump(include={"intents", "entities"})
    loaded = load_bot(write_bot(tmp_path, raw, nlu))
    assert loaded.intents.keys() == b.intents.keys()
    first = loaded.version
    (tmp_path / "bot.yaml").write_text((tmp_path / "bot.yaml").read_text(encoding="utf-8") + "\n# edit\n", encoding="utf-8")
    assert load_bot(tmp_path).version != first


def test_load_bot_reports_problems(tmp_path):
    with pytest.raises(BotError, match="not found"):
        load_bot(tmp_path)
    (tmp_path / "bot.yaml").write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(BotError, match="not valid YAML"):
        load_bot(tmp_path)
    (tmp_path / "bot.yaml").write_text("- a list", encoding="utf-8")
    with pytest.raises(BotError, match="mapping"):
        load_bot(tmp_path)
    write_bot(tmp_path, {"id": "t", "intents": {}}, {"intents": {}})
    with pytest.raises(BotError, match="both bot.yaml and nlu.yaml"):
        load_bot(tmp_path)
    write_bot(tmp_path, {"id": "t"}, {})
    with pytest.raises(BotError, match="scenarios"):
        load_bot(tmp_path)
    raw = bot().model_dump(exclude={"version"})
    raw["scenarios"]["s"]["steps"] = [{"action": "fetch"}]
    write_bot(tmp_path, raw, {})
    with pytest.raises(BotError) as info:
        load_bot(tmp_path, actions=set())
    assert any("unknown action" in p for p in info.value.problems)


def test_yaml_keys_stay_text_and_only_true_false_are_booleans(tmp_path):
    responses = "".join(f"  {k}: {{text: {k}}}\n" for k in REQUIRED) + "  done: {text: 끝}\n"
    (tmp_path / "bot.yaml").write_text(
        "id: t\n"
        "scenarios:\n"
        "  s: {steps: [{respond: done}]}\n"
        "  pick:\n"
        "    steps:\n"
        "      - branch:\n"
        "          on: slots.c\n"
        "          cases: {yes: s, 1: s, true: s, off: s}\n"
        "  other:\n"
        "    slots: {c: {entity: sys.number, ask: done}}\n"
        "    steps: [{collect: c}]\n"
        "responses:\n" + responses,
        encoding="utf-8",
    )
    branch = load_bot(tmp_path).scenarios["pick"].steps[0].branch
    assert branch.on == "slots.c" and set(branch.cases) == {"yes", "1", "true", "off"}


def test_yaml_duplicate_keys_are_errors(tmp_path):
    (tmp_path / "bot.yaml").write_text("id: a\nid: b\n", encoding="utf-8")
    with pytest.raises(BotError, match="duplicate key"):
        load_bot(tmp_path)


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"responses": {**REQUIRED, "fallback": {"text": "{{ slots.c }}?"}, "ask": {"text": "?"}, "done": {"text": "끝"}}},
         "rendered by the engine itself"),
        ({"responses": {**REQUIRED, "ask": {"text": "?"}, "done": {"text": "{{ slots.nope }}"}}}, "unknown variable"),
        ({"responses": {**REQUIRED, "ask": {"text": "{{ slots.c }}?"}, "done": {"text": "끝"}}}, "but also shows it"),
        ({"responses": {**REQUIRED, "ask": {"text": "?"}, "done": {"text": "{{ results.x.y }}"}}}, "unknown variable"),
        ({"scenarios": {"s": {"steps": [{"action": "a", "args": {"q": "{{ slots.nope }}"}}]}}}, "unknown variable"),
        ({"scenarios": {"s": {"steps": [{"branch": {"on": "results.a", "cases": {"1": "s"}}}]}}}, "branch on unknown path"),
        ({"entities": {"city": {"kind": "dictionary", "values": {"서울": ["수도"], "경성": ["수도"]}}}}, "synonym of both"),
        ({"settings": {"max_stack_depth": 2},
          "scenarios": {"s": {"steps": [{"call": "t"}]}, "t": {"steps": [{"call": "u"}]}, "u": {"steps": [{"respond": "done"}]}}},
         "max_stack_depth is 2"),
    ],
)
def test_template_and_structure_validation(over, needle):
    errors = validate(bot(**over)).errors
    assert any(needle in e for e in errors), errors


def test_valid_template_paths_pass():
    b = bot(scenarios={"s": {"slots": {"c": {"entity": "city", "ask": "ask"}},
                             "steps": [{"collect": "c"}, {"action": "w", "args": {"q": "{{ slots.c }}"}},
                                       {"branch": {"on": "results.w.ok", "cases": {"true": "t"}}}, {"respond": "done"}]},
                       "t": {"steps": [{"respond": "done"}]}},
            responses={**REQUIRED, "ask": {"text": "?"}, "done": {"text": "{{ slots.c }} {{ results.w.temp }}"}})
    assert validate(b).errors == []


@pytest.mark.parametrize("q", ["", " ", "x" * 41])
def test_quick_replies_must_be_short_and_non_empty(q):
    with pytest.raises(ValueError):
        bot(responses={**REQUIRED, "ask": {"text": "?", "quick_replies": [q]}, "done": {"text": "끝"}})
