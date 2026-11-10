import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from chatcore.nlu.model import (
    OOS,
    CachedEmbedder,
    IntentClassifier,
    ModelMismatch,
    NluModel,
    Thresholds,
    with_thresholds,
)
from chatcore.nlu.train import (
    CAP_RATIO,
    LEVEL_TARGETS,
    TrainingError,
    _decide_batch,
    _search_thresholds,
    classifier_intents,
    confusable_intents,
    entity_forms,
    evaluate,
    expand,
    fill,
    fit_only,
    format_report,
    main,
    near_duplicate_groups,
    train,
)


def blobs(n_classes, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, 8))
    x = np.vstack([c + 0.3 * rng.normal(size=(10, 8)) for c in centers]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels = [f"c{i}" for i in range(n_classes) for _ in range(10)]
    return x, labels


@pytest.mark.parametrize("n", [2, 4])
def test_classifier_matches_sklearn_predict_proba(n):
    x, labels = blobs(n)
    labels = [OOS if lab == "c0" else lab for lab in labels]
    clf = IntentClassifier.fit(x, labels)
    ref = LogisticRegression(C=10.0, max_iter=5000, class_weight="balanced", random_state=0).fit(x, labels)
    probs, sims = clf.predict(x)
    np.testing.assert_allclose(probs, ref.predict_proba(x), atol=1e-5)
    assert list(clf.classes) == list(ref.classes_)
    in_scope = np.array([lab != OOS for lab in labels])
    np.testing.assert_allclose(sims[in_scope], 1.0, atol=1e-5)  # in-scope rows are in the index


def test_classifier_save_load_round_trip(tmp_path):
    x, labels = blobs(3)
    clf = IntentClassifier.fit(x, labels)
    clf.save(tmp_path / "c.npz")
    loaded = IntentClassifier.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(loaded.predict(x)[0], clf.predict(x)[0])


@pytest.mark.parametrize(
    "x, labels, msg",
    [(np.zeros((0, 4), np.float32), [], "at least one row"), (np.ones((2, 4), np.float32), ["a", "a"], "two intents"),
     (np.ones((2, 4), np.float32), ["a"], "one label per"), (np.eye(2, dtype=np.float32), [OOS, OOS], "two intents")],
)
def test_classifier_fit_errors(x, labels, msg):
    with pytest.raises(ValueError, match=msg):
        IntentClassifier.fit(x, labels)


def test_expand_is_capped_deterministic_and_fills_dictionary_entities(sample_bot):
    forms = entity_forms(sample_bot)
    templates = ["{location} 날씨", "{location} 비 와?", "날씨 어때"]
    rows = expand(templates, forms, seed=1)
    assert rows == expand(templates, forms, seed=1)
    assert len(rows) <= CAP_RATIO * len(templates) and "날씨 어때" in rows
    assert all("{" not in r for r in rows)
    assert expand([], forms) == []


def test_fill_rejects_unknown_placeholder(sample_bot):
    import random

    with pytest.raises(TrainingError, match="not a dictionary entity"):
        fill("{nope} 날씨", random.Random(0), entity_forms(sample_bot))


def test_near_duplicate_groups():
    g = near_duplicate_groups(["{location} 날씨 알려줘", "{location} 날씨 알려줘요", "지금 몇 시야"])
    assert g[0] == g[1] != g[2] and near_duplicate_groups([]) == []


def test_evaluate_and_train_save_a_loadable_model(sample_bot, fake_embedder, extractor, tmp_path):
    emb = CachedEmbedder(fake_embedder)
    directory, report = train(sample_bot, emb, tmp_path, fresh=fake_embedder)
    assert directory == tmp_path / "sample" / sample_bot.version
    assert 0 < report.macro_f1 <= 1 and set(report.per_intent_f1) >= set(sample_bot.intents)
    assert report.determinism_max_diff == 0.0
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["bot_version"] == sample_bot.version and meta["embedding_model"] == "fake-ngram-256"
    model = NluModel.load(sample_bot, directory, emb, extractor)
    assert model.parse("안녕하세요").matched_by == "exact"
    assert "macro-F1" in format_report(report, directory)


def test_model_load_rejects_mismatches(sample_bot, fake_embedder, extractor, tmp_path):
    directory, _ = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    other = sample_bot.model_copy(update={"version": "other"})
    with pytest.raises(ModelMismatch, match="make train"):
        NluModel.load(other, directory, fake_embedder, extractor)

    class Other:
        model_id = "another-model"
        prompt = ""

        def encode(self, texts):
            raise AssertionError

    with pytest.raises(ModelMismatch, match="trained with"):
        NluModel.load(sample_bot, directory, Other(), extractor)


def test_evaluate_needs_enough_examples(sample_bot, fake_embedder):
    tiny = sample_bot.model_copy(update={"intents": {k: v.model_copy(update={"examples": v.examples[:2]})
                                                      for k, v in sample_bot.intents.items()}})
    with pytest.raises(TrainingError, match="at least 5"):
        evaluate(tiny, fake_embedder)


def test_parse_is_deterministic(sample_bot, fake_embedder, extractor, tmp_path):
    directory, _ = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    model = NluModel.load(sample_bot, directory, fake_embedder, extractor)
    assert model.parse("서울 날씨 좀 알려 주라") == model.parse("서울 날씨 좀 알려 주라")


def test_cli_exit_codes(sample_bot, fake_embedder, tmp_path, monkeypatch, capsys):
    import chatcore.nlu.train as train_mod

    monkeypatch.setattr(train_mod, "SentenceEmbedder", lambda config=None: fake_embedder)
    bot_dir = __import__("pathlib").Path(__file__).resolve().parents[3] / "bots/sample"
    assert main([str(bot_dir), "--out", str(tmp_path), "--min-f1", "0"]) == 0
    assert "per-intent F1" in capsys.readouterr().out
    out2 = tmp_path / "gated"
    assert main([str(bot_dir), "--out", str(out2), "--min-f1", "1.01"]) == 1
    assert not out2.exists()  # a model below the bar is not saved
    assert "model not saved" in capsys.readouterr().err
    assert main([str(tmp_path / "missing"), "--out", str(tmp_path)]) == 2


def test_decide_mirrors_interpreter_rules(sample_bot):
    from chatcore.nlu.train import CLARIFY, decide

    classes = ["out_of_scope", "temperature", "weather"]
    probs = np.array([[0.1, 0.1, 0.8], [0.1, 0.42, 0.48], [0.1, 0.05, 0.85], [0.7, 0.1, 0.2], [0.05, 0.05, 0.25]])
    sims = np.array([0.9, 0.9, 0.2, 0.9, 0.9])
    assert decide(probs, sims, classes, sample_bot) == ["weather", CLARIFY, OOS, OOS, OOS]


def test_parse_embeds_normalized_text(sample_bot, fake_embedder, extractor, tmp_path):
    directory, _ = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    model = NluModel.load(sample_bot, directory, fake_embedder, extractor)
    model.parse("ＡＢＣ   날씨는\u3000어때")
    assert fake_embedder.last == ["ABC 날씨는 어때"]


def test_model_load_needs_files_and_matching_prompt(sample_bot, fake_embedder, extractor, tmp_path):
    with pytest.raises(ModelMismatch, match="no trained model"):
        NluModel.load(sample_bot, tmp_path, fake_embedder, extractor)
    directory, _ = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path)

    class Prompted:
        model_id = "fake-ngram-256"
        prompt = "query: "

        def encode(self, texts):
            raise AssertionError

    with pytest.raises(ModelMismatch, match="prompt"):
        NluModel.load(sample_bot, directory, Prompted(), extractor)


def test_training_the_same_version_again_swaps_in_complete_files(sample_bot, fake_embedder, extractor, tmp_path, monkeypatch):
    emb = CachedEmbedder(fake_embedder)
    directory, _ = train(sample_bot, emb, tmp_path)
    first = json.loads((directory / "meta.json").read_text(encoding="utf-8"))["created_at"]
    again, _ = train(sample_bot, emb, tmp_path)
    assert again == directory and sorted(p.name for p in directory.parent.iterdir()) == [sample_bot.version]
    assert json.loads((directory / "meta.json").read_text(encoding="utf-8"))["created_at"] >= first
    NluModel.load(sample_bot, directory, emb, extractor)

    def broken_save(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(IntentClassifier, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        train(sample_bot, emb, tmp_path)
    assert sorted(p.name for p in directory.parent.iterdir()) == [sample_bot.version]  # no half-written leftovers
    NluModel.load(sample_bot, directory, emb, extractor)  # the previous model is untouched


def test_fit_only_needs_no_evaluation_but_two_intents(sample_bot, fake_embedder, extractor, tmp_path):
    from chatcore.nlu.train import TrainingError, fit_only

    tiny = sample_bot.model_copy(update={"intents": {
        "greeting": sample_bot.intents["greeting"].model_copy(update={"examples": ["안녕"]}),
        "weather": sample_bot.intents["weather"].model_copy(update={"examples": ["날씨 어때"]})}})
    directory = fit_only(tiny, CachedEmbedder(fake_embedder), tmp_path)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["test_build"] is True and meta["macro_f1"] is None
    assert NluModel.load(tiny, directory, fake_embedder, extractor).parse("날씨 어때").intents[0].intent == "weather"
    alone = sample_bot.model_copy(update={"intents": {"weather": sample_bot.intents["weather"]}})
    with pytest.raises(TrainingError, match="two intents"):
        fit_only(alone, CachedEmbedder(fake_embedder), tmp_path)


def test_classifier_data_errors_become_training_errors(sample_bot, fake_embedder, tmp_path, monkeypatch):
    from chatcore.nlu.train import TrainingError, fit_only

    def refuse(*args, **kwargs):
        raise ValueError("need at least one in-scope example")

    monkeypatch.setattr(IntentClassifier, "fit", refuse)
    with pytest.raises(TrainingError, match="in-scope"):
        fit_only(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    assert not list(tmp_path.iterdir())  # nothing half-saved


# ---------------------------------------------------------------- E13: exact_only intents


def test_classifier_intents_excludes_exact_only(sample_bot):
    marked = sample_bot.model_copy(update={"intents": {
        **sample_bot.intents, "thanks": sample_bot.intents["thanks"].model_copy(update={"exact_only": True})}})
    intents = classifier_intents(marked)
    assert "thanks" not in intents and set(intents) == set(sample_bot.intents) - {"thanks"}


def test_evaluate_excludes_exact_only_intents_from_cv_and_the_few_examples_check(sample_bot, fake_embedder):
    marked = sample_bot.model_copy(update={"intents": {
        **sample_bot.intents, "thanks": sample_bot.intents["thanks"].model_copy(
            update={"examples": ["고마워"], "exact_only": True})}})
    report = evaluate(marked, fake_embedder)  # a non-exact_only intent with 1 example would raise "at least 5"
    assert "thanks" not in report.per_intent_f1


def test_fit_only_errors_when_exact_only_leaves_fewer_than_two_classifier_intents(sample_bot, fake_embedder, tmp_path):
    one_left = sample_bot.model_copy(update={"intents": {
        "weather": sample_bot.intents["weather"],
        "greeting": sample_bot.intents["greeting"].model_copy(update={"exact_only": True}),
    }})
    with pytest.raises(TrainingError, match="two intents"):
        fit_only(one_left, CachedEmbedder(fake_embedder), tmp_path)


def test_exact_only_intent_still_matches_exactly(sample_bot, fake_embedder, extractor, tmp_path):
    marked = sample_bot.model_copy(update={"intents": {
        **sample_bot.intents, "thanks": sample_bot.intents["thanks"].model_copy(update={"exact_only": True})}})
    directory, _ = train(marked, CachedEmbedder(fake_embedder), tmp_path)
    model = NluModel.load(marked, directory, fake_embedder, extractor)
    literal = next(ex for ex in marked.intents["thanks"].examples if "{" not in ex)
    result = model.parse(literal)
    assert result.matched_by == "exact" and result.intents[0].intent == "thanks"


# ---------------------------------------------------------------- E13: confusable intents


def test_confusable_intents_finds_identical_examples_and_sorts_by_similarity(sample_bot, fake_embedder):
    bot = sample_bot.model_copy(update={"intents": {
        "greeting": sample_bot.intents["greeting"].model_copy(update={"examples": ["안녕하세요"]}),
        "weather": sample_bot.intents["weather"].model_copy(update={"examples": ["안녕하세요"]}),
        "bye": sample_bot.intents["bye"].model_copy(update={"examples": ["완전히 다른 문장입니다 그리고 아주 길다"]}),
        "out_of_scope": sample_bot.intents["out_of_scope"],
    }})
    pairs = confusable_intents(bot, fake_embedder, threshold=0.999)
    assert len(pairs) == 1 and (pairs[0].a, pairs[0].b) == ("greeting", "weather")
    assert pairs[0].similarity == pytest.approx(1.0) and (pairs[0].example_a, pairs[0].example_b) == ("안녕하세요", "안녕하세요")


def test_confusable_intents_needs_at_least_two_in_scope_intents(sample_bot, fake_embedder):
    one = sample_bot.model_copy(update={"intents": {"weather": sample_bot.intents["weather"]}})
    assert confusable_intents(one, fake_embedder) == []


def test_train_report_and_meta_include_confusable_pairs(sample_bot, fake_embedder, tmp_path):
    directory, report = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    assert isinstance(report.confusable, list)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert "confusable" in meta and "thresholds" in meta


# ---------------------------------------------------------------- E13: similarity_level search


def test_search_thresholds_tie_break_prefers_the_lowest_thresholds():
    classes = (OOS, "a")
    probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # OOS always rejected, "a" always accepted
    sims = np.array([0.0, 1.0], dtype=np.float32)
    thresholds, warning = _search_thresholds([(probs, sims, classes, [OOS, "a"])], "low", ambiguity_margin=0.1)
    assert warning is None
    assert (thresholds.intent_threshold, thresholds.similarity_threshold) == (0.0, 0.0)


def test_search_thresholds_meets_the_target_without_hurting_in_scope_recall():
    classes = (OOS, "a", "b")
    probs = np.array([
        [0.1, 0.6, 0.3],    # OOS row the classifier is fooled by, at 0.6
        [0.2, 0.5, 0.3],    # OOS row the classifier is fooled by, at 0.5
        [0.1, 0.8, 0.1],    # true "a", confidently
        [0.05, 0.15, 0.8],  # true "b", confidently
    ], dtype=np.float32)
    sims = np.array([0.5, 0.3, 0.9, 0.85], dtype=np.float32)
    true = [OOS, OOS, "a", "b"]
    folds = [(probs, sims, classes, true)]
    thresholds, warning = _search_thresholds(folds, "normal", ambiguity_margin=0.1)  # target 0.90
    assert warning is None and thresholds.level == "normal"
    y_pred = _decide_batch(probs, sims, classes, thresholds.intent_threshold, thresholds.similarity_threshold, 0.1)
    oos_rows = [p for t, p in zip(true, y_pred, strict=True) if t == OOS]
    in_rows = [(t, p) for t, p in zip(true, y_pred, strict=True) if t != OOS]
    assert sum(p == OOS for p in oos_rows) / len(oos_rows) >= LEVEL_TARGETS["normal"]
    assert sum(p == t for t, p in in_rows) / len(in_rows) == 1.0  # both in-scope rows still decided correctly


def test_search_thresholds_falls_back_with_a_warning_when_the_target_is_unreachable():
    classes = (OOS, "a")
    # neither row can ever be rejected: probability and similarity are both above the whole grid
    probs = np.array([[0.05, 0.95], [0.05, 0.95]], dtype=np.float32)
    sims = np.array([0.96, 0.96], dtype=np.float32)
    thresholds, warning = _search_thresholds([(probs, sims, classes, [OOS, "a"])], "very_high", ambiguity_margin=0.1)
    assert warning is not None and "very_high" in warning
    assert thresholds.level == "very_high"


def test_evaluate_with_similarity_level_but_no_out_of_scope_examples_warns_and_keeps_document_numbers(
        sample_bot, fake_embedder):
    no_oos = sample_bot.model_copy(update={
        "settings": sample_bot.settings.model_copy(update={"similarity_level": "high"}),
        "intents": {k: v for k, v in sample_bot.intents.items() if k != OOS},
    })
    report = evaluate(no_oos, fake_embedder)
    assert report.thresholds.level is None
    assert report.thresholds.intent_threshold == sample_bot.settings.intent_threshold
    assert any("no out_of_scope examples" in w for w in report.warnings)


def test_evaluate_with_similarity_level_searches_thresholds(sample_bot, fake_embedder):
    leveled = sample_bot.model_copy(
        update={"settings": sample_bot.settings.model_copy(update={"similarity_level": "low"})})
    report = evaluate(leveled, fake_embedder)
    assert report.thresholds.level == "low"
    assert report.oos_recall is not None and report.in_scope_recall is not None
    if not any("targets out-of-scope rejection" in w for w in report.warnings):
        assert report.oos_recall >= LEVEL_TARGETS["low"]


def test_similarity_levels_are_monotonic_in_rejection_and_recall(sample_bot, fake_embedder):
    """Raising the level never lowers the achieved out-of-scope rejection and never raises
    in-scope recall (docs/engine-core-plan.md "빌더 단계 E13")."""
    results = []
    for level in ("low", "normal", "somewhat_high", "high", "very_high"):
        leveled = sample_bot.model_copy(
            update={"settings": sample_bot.settings.model_copy(update={"similarity_level": level})})
        report = evaluate(leveled, fake_embedder)
        results.append((report.oos_recall, report.in_scope_recall))
    rejections = [r for r, _ in results]
    recalls = [c for _, c in results]
    assert all(a <= b + 1e-9 for a, b in zip(rejections[:-1], rejections[1:], strict=True))
    assert all(a >= b - 1e-9 for a, b in zip(recalls[:-1], recalls[1:], strict=True))


def test_fit_only_meta_records_document_thresholds_with_a_null_level(sample_bot, fake_embedder, tmp_path):
    directory = fit_only(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    s = sample_bot.settings
    assert meta["thresholds"] == {"intent_threshold": s.intent_threshold, "similarity_threshold": s.similarity_threshold,
                                  "ambiguity_margin": s.ambiguity_margin, "level": None}


# ---------------------------------------------------------------- E13: thresholds applied at load


def test_with_thresholds_changes_settings_but_keeps_the_bot_version(sample_bot):
    th = Thresholds(0.5, 0.6, 0.2, "high")
    served = with_thresholds(sample_bot, th)
    assert served.version == sample_bot.version
    assert (served.settings.intent_threshold, served.settings.similarity_threshold,
            served.settings.ambiguity_margin) == (0.5, 0.6, 0.2)
    assert served.settings.rerun_window == sample_bot.settings.rerun_window  # untouched


def test_nlu_model_thresholds_reads_meta_or_falls_back_to_the_document(sample_bot):
    raw = {"thresholds": {"intent_threshold": 0.5, "similarity_threshold": 0.6, "ambiguity_margin": 0.2,
                          "level": "high"}}
    assert NluModel.thresholds(raw, sample_bot) == Thresholds(0.5, 0.6, 0.2, "high")
    s = sample_bot.settings
    assert NluModel.thresholds({}, sample_bot) == Thresholds(s.intent_threshold, s.similarity_threshold,
                                                              s.ambiguity_margin, None)


def test_nlu_model_load_applies_the_stored_thresholds(sample_bot, fake_embedder, extractor, tmp_path):
    directory, _ = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["thresholds"] = {"intent_threshold": 0.77, "similarity_threshold": 0.11, "ambiguity_margin": 0.05,
                          "level": "very_high"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    model = NluModel.load(sample_bot, directory, fake_embedder, extractor)
    assert (model.bot.settings.intent_threshold, model.bot.settings.similarity_threshold,
            model.bot.settings.ambiguity_margin) == (0.77, 0.11, 0.05)
    assert model.bot.version == sample_bot.version


def test_nlu_model_load_without_a_thresholds_key_uses_the_document_numbers(sample_bot, fake_embedder, extractor, tmp_path):
    """An older training's meta.json, saved before this field existed."""
    directory, _ = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["thresholds"]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    model = NluModel.load(sample_bot, directory, fake_embedder, extractor)
    assert model.bot.settings.intent_threshold == sample_bot.settings.intent_threshold
    assert model.bot.settings.similarity_threshold == sample_bot.settings.similarity_threshold


@pytest.mark.parametrize("thresholds", [{"intent_threshold": 0.3}, {"intent_threshold": "x", "similarity_threshold": 0.4,
                                                                   "ambiguity_margin": 0.1}, "broken"])
def test_a_malformed_thresholds_entry_in_meta_is_a_model_mismatch(sample_bot, fake_embedder, tmp_path, thresholds):
    directory = fit_only(sample_bot, CachedEmbedder(fake_embedder), tmp_path)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    meta["thresholds"] = thresholds
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if isinstance(thresholds, dict):
        with pytest.raises(ModelMismatch, match="thresholds"):
            NluModel.load(sample_bot, directory, fake_embedder)
    else:  # not even a mapping: treated like an older model without stored thresholds
        assert NluModel.load(sample_bot, directory, fake_embedder).bot.settings == sample_bot.settings
