"""Phase 0 one-time migration: legacy Rasa NLU data -> new bot NLU format.

Reads rasaServer/data/nlu.yml and the legacy LOCATION_KEYWORDS array in
SimpleSolver.java, and writes bots/sample/nlu.yaml:

- `[부산](location)` annotations become `{location}` placeholders and the values
  feed the `location` dictionary entity (canonical value + synonyms).
- Deictic "locations" (여기, 그 동네, ...) are not location values; their
  annotation is dropped and the words stay as plain text.
- `unknown_location` examples are merged into weather / temperature.
- Exact duplicates are removed; cross-intent conflicts are resolved by the
  explicit tables below and reported. Anything unresolved is an error.

Usage: uv run python bench/convert_rasa_nlu.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RASA_NLU = ROOT / "rasaServer/data/nlu.yml"
SOLVER_JAVA = ROOT / "src/main/java/com/yk/chatbot/lasa/impl/SimpleSolver.java"
OOS_TRAIN = ROOT / "bench/data/out_of_scope_train.yaml"
OOS_EVAL = ROOT / "bench/data/out_of_scope_eval.yaml"
ADDITIONS = ROOT / "bench/data/additions.yaml"
OUTPUT = ROOT / "bots/sample/nlu.yaml"

ANNOTATION = re.compile(r"\[([^\]]+)\]\(([a-z_]+)\)")
SUPPORTED_ENTITIES = {"location"}

DEICTIC_LOCATIONS = frozenset(
    {"거기", "그 동네", "그곳", "그쪽", "어디", "어딘가", "여기", "우리 동네",
     "이 근처", "이 지역", "저기", "지역", "해당 지역"}
)

# Province full names whose short form is not a plain suffix strip.
PROVINCE_SHORT = {
    "경상남도": "경남", "경상북도": "경북", "전라남도": "전남", "전라북도": "전북",
    "충청남도": "충남", "충청북도": "충북",
}
# Longest first so "특별자치시" wins over "시".
ADMIN_SUFFIXES = ("특별자치시", "특별자치도", "특별시", "광역시", "시", "도")

# Legacy data errors, resolved by hand (each one is printed as a note on run).
CROSS_INTENT_RESOLUTION = {"안녕": "greeting"}  # was in greeting AND bye
RELABEL = {
    ("greeting", "도움이 필요해요"): "help",  # a help request filed under greeting
    ("greeting", "사용법 알려주세요"): "help",
    ("greeting", "좋은 하루 되세요"): "bye",  # farewells filed under greeting
    ("greeting", "좋은 밤 되세요"): "bye",
    ("bye", "상담 감사합니다"): "thanks",
    ("compliment", "항상 정확한 정보 고마워"): "thanks",
}
DROP = {
    ("greeting", "오늘 날씨 좋네요"),  # weather small talk; collides with the weather intent
    ("greeting", "상담원 연결해주세요"),  # human hand-off request; no such intent yet
}
# Templates that collapse to the same text in several intents → the one owner.
TEMPLATE_RESOLUTION = {"{location}는 지금 몇 도에요?": "temperature"}  # a 몇 도 question filed under weather

TEMPERATURE_WORDS = re.compile(r"기온|온도|몇 ?도")

INTENT_LABELS = {
    "greeting": "인사", "weather": "날씨 문의", "temperature": "기온 문의",
    "time": "현재 시각", "bye": "작별 인사", "thanks": "감사", "help": "도움말",
    "compliment": "칭찬", "negative_feedback": "불만·지적",
    "out_of_scope": "범위 밖",
}


class ConversionError(ValueError):
    pass


@dataclass
class Conversion:
    intents: dict[str, list[str]] = field(default_factory=dict)
    location_values: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def to_template(text: str) -> tuple[str, list[str]]:
    """Replace location annotations with `{location}`; return (template, values).

    Deictic annotations are unwrapped to plain text. Unsupported entity types
    raise, so new legacy annotations cannot be silently dropped.
    """
    values: list[str] = []

    def repl(m: re.Match[str]) -> str:
        value, entity = m.group(1).strip(), m.group(2)
        if entity not in SUPPORTED_ENTITIES:
            raise ConversionError(f"unsupported entity type {entity!r} in {text!r}")
        if not value:
            raise ConversionError(f"empty entity value in {text!r}")
        if value in DEICTIC_LOCATIONS:
            return value
        values.append(value)
        return "{location}"

    template = ANNOTATION.sub(repl, text).strip()
    if not template:
        raise ConversionError("empty example")
    return template, values


def read_rasa_examples(path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not data or not data.get("nlu"):
        raise ConversionError(f"{path} has no 'nlu' section")
    rows: list[tuple[str, str]] = []
    for block in data["nlu"]:
        intent = block.get("intent")
        if not intent:
            raise ConversionError(f"non-intent block not supported: {sorted(block)}")
        lines = [ln.strip() for ln in (block.get("examples") or "").splitlines() if ln.strip()]
        odd = [ln for ln in lines if not ln.startswith("- ")]
        if odd:
            raise ConversionError(f"intent {intent!r} has lines that are not '- ' examples: {odd}")
        examples = [ln[2:].strip() for ln in lines]
        if not examples:
            raise ConversionError(f"intent {intent!r} has no examples")
        rows.extend((intent, ex) for ex in examples)
    return rows


def read_legacy_keywords(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    m = re.search(r"LOCATION_KEYWORDS\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        raise ConversionError(f"LOCATION_KEYWORDS not found in {path}")
    keywords = re.findall(r'"([^"]+)"', m.group(1))
    if not keywords:
        raise ConversionError("LOCATION_KEYWORDS is empty")
    return list(dict.fromkeys(keywords))  # dedupe, keep order


def convert(rows: list[tuple[str, str]], legacy_keywords: list[str]) -> Conversion:
    conv = Conversion()
    seen_text: dict[str, str] = {}
    for intent, text in rows:
        if (intent, text) in DROP:
            conv.notes.append(f"drop from {intent}: {text}")
            continue
        if intent == "unknown_location":
            target = "temperature" if TEMPERATURE_WORDS.search(text) else "weather"
            conv.notes.append(f"unknown_location → {target}: {text}")
            intent = target
        if (intent, text) in RELABEL:
            new = RELABEL[(intent, text)]
            conv.notes.append(f"relabel {intent} → {new}: {text}")
            intent = new
        if text in CROSS_INTENT_RESOLUTION:
            keep = CROSS_INTENT_RESOLUTION[text]
            if intent != keep:
                conv.notes.append(f"drop conflicting example from {intent} (kept in {keep}): {text}")
                continue
        elif text in seen_text and seen_text[text] != intent:
            raise ConversionError(
                f"example {text!r} is in both {seen_text[text]!r} and {intent!r}; "
                "add it to CROSS_INTENT_RESOLUTION"
            )
        seen_text[text] = intent

        template, values = to_template(text)
        conv.location_values.update(values)
        bucket = conv.intents.setdefault(intent, [])
        if template not in bucket:
            bucket.append(template)
    owners: dict[str, list[str]] = {}
    for intent, templates in conv.intents.items():
        for template in templates:
            owners.setdefault(template, []).append(intent)
    for template, keep in TEMPLATE_RESOLUTION.items():
        holders = owners.get(template, [])
        if not holders:
            continue
        for intent in holders:
            if intent != keep:
                conv.intents[intent].remove(template)
                conv.notes.append(f"move template {intent} → {keep}: {template}")
        if keep not in holders:
            conv.intents.setdefault(keep, []).append(template)
        owners[template] = [keep]
    clashes = {t: i for t, i in owners.items() if len(i) > 1}
    if clashes:
        raise ConversionError(f"templates shared by several intents: {clashes}")
    emptied = [i for i, ts in conv.intents.items() if not ts]
    if emptied:
        raise ConversionError(f"intents left without examples: {emptied}")
    conv.location_values.update(legacy_keywords)
    return conv


def group_synonyms(values: set[str]) -> dict[str, list[str]]:
    """Map each canonical location to its synonyms (full administrative names)."""
    canonical_of: dict[str, str] = {}
    for v in values:
        if v in PROVINCE_SHORT:
            canonical_of[v] = PROVINCE_SHORT[v]
            continue
        canonical_of[v] = v
        for suffix in ADMIN_SUFFIXES:
            base = v[: -len(suffix)]
            if v.endswith(suffix) and len(base) >= 2 and " " not in v:
                # "도" only when the short form is itself a known value (강원도 → 강원),
                # so 전라도 stays its own value.
                if suffix == "도" and base not in values:
                    break
                canonical_of[v] = base
                break
    groups: dict[str, list[str]] = {}
    for v, c in canonical_of.items():
        groups.setdefault(c, [])
        if v != c:
            groups[c].append(v)
    return {c: sorted(s) for c, s in sorted(groups.items())}


def load_oos_examples(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    examples = [str(e).strip() for e in data.get("examples") or []]
    if not examples or any(not e for e in examples):
        raise ConversionError(f"{path} must list non-empty 'examples'")
    return list(dict.fromkeys(examples))


def load_additions(path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    additions = {str(k): [str(e).strip() for e in v or []] for k, v in (data.get("intents") or {}).items()}
    bad = [k for k, v in additions.items() if not v or any(not e for e in v)]
    if not additions or bad:
        raise ConversionError(f"{path} must map intents to non-empty example lists (bad: {bad})")
    return additions


def merge_additions(conv: Conversion, additions: dict[str, list[str]]) -> None:
    """Append hand-written examples; each must be new across ALL intents."""
    existing = {e for exs in conv.intents.values() for e in exs}
    for intent, examples in additions.items():
        if intent not in conv.intents:
            raise ConversionError(f"additions target unknown intent {intent!r}")
        for e in examples:
            if e in existing:
                raise ConversionError(f"addition {e!r} ({intent}) already exists")
            existing.add(e)
            conv.intents[intent].append(e)


def build_document(conv: Conversion, oos: list[str], oos_eval: list[str] = ()) -> dict:
    overlap = {e for exs in conv.intents.values() for e in exs} & set(oos)
    if overlap:
        raise ConversionError(f"out_of_scope examples overlap intents: {sorted(overlap)}")
    intents = dict(conv.intents)
    intents["out_of_scope"] = oos
    leaked = {e for exs in intents.values() for e in exs} & set(oos_eval)
    if leaked:
        raise ConversionError(f"held-out evaluation sentences found in training data: {sorted(leaked)}")
    missing = set(intents) - set(INTENT_LABELS)
    if missing:
        raise ConversionError(f"no label for intents: {sorted(missing)}")
    return {
        "intents": {
            name: {"label": INTENT_LABELS[name], "examples": exs}
            for name, exs in intents.items()
        },
        "entities": {
            "location": {"kind": "dictionary", "values": group_synonyms(conv.location_values)}
        },
    }


HEADER = (
    "# Generated by bench/convert_rasa_nlu.py from rasaServer/data/nlu.yml (+ legacy\n"
    "# LOCATION_KEYWORDS, bench/data/additions.yaml, bench/data/out_of_scope_train.yaml).\n"
)


def main() -> int:
    rows = read_rasa_examples(RASA_NLU)
    conv = convert(rows, read_legacy_keywords(SOLVER_JAVA))
    merge_additions(conv, load_additions(ADDITIONS))
    doc = build_document(conv, load_oos_examples(OOS_TRAIN), load_oos_examples(OOS_EVAL))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        HEADER + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )
    print(f"read {len(rows)} legacy examples → {OUTPUT.relative_to(ROOT)}")
    for name, spec in doc["intents"].items():
        print(f"  {name:18} {len(spec['examples']):4} templates")
    loc = doc["entities"]["location"]["values"]
    print(f"  location values: {len(loc)} canonical, {sum(map(len, loc.values()))} synonyms")
    for note in conv.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
