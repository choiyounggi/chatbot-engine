import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from chatcore.nlu.model import OOS, CachedEmbedder, IntentClassifier, ModelMismatch, NluModel
from chatcore.nlu.train import (
    CAP_RATIO,
    TrainingError,
    entity_forms,
    evaluate,
    expand,
    fill,
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
