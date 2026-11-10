
from chatcore.nlu.keyword import KeywordMatcher
from chatcore.nlu.model import CachedEmbedder, NluModel
from chatcore.nlu.train import fit_only

REQUIRED = {k: {"text": k} for k in ("fallback", "cancelled", "restarted", "error", "clarify", "input_invalid")}


def _kwbot(sample_bot, keyword_match: bool):
    from chatcore.bot.schema import BotDefinition

    data = {
        "id": "kw", "version": "v",
        "settings": {"keyword_match": keyword_match, "similarity_threshold": 0.0},
        "intents": {
            "greeting": {"examples": ["안녕하세요"]},
            "weather": {"examples": ["오늘 날씨 알려줘"]},
            "out_of_scope": {"examples": ["아무 상관 없는 말이에요"]},
        },
        "scenarios": {"s": {"trigger": {"intent": "weather"}, "steps": [{"respond": "done"}]}},
        "responses": {**REQUIRED, "done": {"text": "끝"}},
    }
    return BotDefinition.model_validate(data)


def test_morphemes_drops_particles_and_endings_keeping_the_verb_stem():
    m = KeywordMatcher()
    assert m.morphemes("오늘 날씨 알려줘") == m.morphemes("오늘 날씨 알려줬어요")
    assert m.morphemes("부산 날씨 알려줘") == m.morphemes("부산은 날씨 알려줘요")
    assert m.morphemes("고마워") == m.morphemes("고마워요") == ("고맙",)


def test_morphemes_distinguishes_different_content():
    m = KeywordMatcher()
    assert m.morphemes("오늘 날씨 알려줘") != m.morphemes("내일 날씨 알려줘")
    assert m.morphemes("그거") == ("그거",)


def test_morphemes_of_an_empty_or_punctuation_only_input_is_empty():
    m = KeywordMatcher()
    assert m.morphemes("") == ()
    assert m.morphemes("...") == ()


def test_keyword_match_catches_a_particle_and_ending_variant_when_enabled(sample_bot, fake_embedder, tmp_path):
    bot = _kwbot(sample_bot, keyword_match=True)
    directory = fit_only(bot, CachedEmbedder(fake_embedder), tmp_path)
    model = NluModel.load(bot, directory, fake_embedder)
    result = model.parse("오늘 날씨 알려줬어요")  # not a literal example, but the same core morphemes
    assert result.matched_by == "keyword" and result.intents[0].intent == "weather" and result.intents[0].score == 1.0


def test_keyword_match_off_by_default_falls_through_to_the_classifier(sample_bot, fake_embedder, tmp_path):
    bot = _kwbot(sample_bot, keyword_match=False)
    directory = fit_only(bot, CachedEmbedder(fake_embedder), tmp_path)
    model = NluModel.load(bot, directory, fake_embedder)
    result = model.parse("오늘 날씨 알려줬어요")
    assert result.matched_by == "classifier"


def test_keyword_match_does_not_match_unrelated_input(sample_bot, fake_embedder, tmp_path):
    bot = _kwbot(sample_bot, keyword_match=True)
    directory = fit_only(bot, CachedEmbedder(fake_embedder), tmp_path)
    model = NluModel.load(bot, directory, fake_embedder)
    result = model.parse("그거")
    assert result.matched_by == "classifier"


def test_a_keyword_key_two_intents_share_matches_neither(sample_bot, fake_embedder, tmp_path):
    # "배송 조회해요" and "배송 조회할래요" both reduce to (배송, 조회): which intent is meant is unknown
    from chatcore.bot.schema import BotDefinition
    from chatcore.nlu.train import keyword_collisions

    data = _kwbot(sample_bot, keyword_match=True).model_dump()
    data["intents"]["track"] = {"examples": ["배송 조회해요"]}
    data["intents"]["cancel_order"] = {"examples": ["배송 조회할래요"]}
    bot = BotDefinition.model_validate(data)
    directory = fit_only(bot, CachedEmbedder(fake_embedder), tmp_path)
    model = NluModel.load(bot, directory, fake_embedder)
    assert model.parse("배송 조회해요").matched_by == "exact"  # the literal sentence still matches exactly
    assert model.parse("배송 조회했어요").matched_by == "classifier"  # the shared key does not decide
    assert keyword_collisions(bot) == [("배송 조회", ["cancel_order", "track"])]
    assert keyword_collisions(_kwbot(sample_bot, keyword_match=False)) == []
