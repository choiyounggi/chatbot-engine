import random

import numpy as np
import pytest
from kiwipiepy import Kiwi

from bench.phase0 import (
    CAP_RATIO,
    ENT_TOKEN,
    OOS,
    BenchError,
    Delexicalizer,
    IntentModel,
    Setup,
    choose_threshold,
    clarify_stats,
    combine_groups,
    decide,
    evaluate,
    expand,
    fill,
    macro_f1_idx,
    near_duplicate_groups,
    normalize,
    oos_recall,
    paired_bootstrap,
    prepare,
    recommend,
    semantic_groups,
    split_halves,
    template_groups,
    training_rows,
    wilson,
)

FORMS = sorted(["부산광역시", "부산", "서울"], key=lambda f: (-len(f), f))


@pytest.fixture(scope="module")
def kiwi():
    return Kiwi()


@pytest.fixture(scope="module")
def delex(kiwi):
    return Delexicalizer(FORMS, kiwi)


def test_normalize_collapses_whitespace_and_nfkc():
    assert normalize("  날씨\t\n어때 ") == "날씨 어때"
    assert normalize("ＡＢＣ") == "ABC"


def test_fill_replaces_every_placeholder():
    out = fill("{location} 말고 {location}", random.Random(0), FORMS)
    assert "{location}" not in out and out.count(" ") == 2


def test_fill_without_values_raises():
    with pytest.raises(BenchError):
        fill("{location} 날씨", random.Random(0), [])


def test_expand_respects_cap_and_keeps_every_template():
    templates = ["{location} 날씨", "{location} 비 와?", "날씨 어때"]
    rows = expand(templates, FORMS)
    assert len(rows) <= CAP_RATIO * len(templates)
    assert "날씨 어때" in rows
    assert any(r.endswith(" 날씨") for r in rows) and any(r.endswith(" 비 와?") for r in rows)


def test_expand_is_deterministic():
    assert expand(["{location} 날씨"], FORMS, seed=3) == expand(["{location} 날씨"], FORMS, seed=3)


def test_expand_empty_intent_gives_no_rows():
    assert expand([], FORMS) == []


def test_near_duplicate_groups_cluster_rephrasings_only():
    groups = near_duplicate_groups(["{location} 날씨 알려줘", "{location} 날씨 알려줘요", "지금 몇 시야"])
    assert groups[0] == groups[1] != groups[2]


def test_near_duplicate_groups_edge_cases():
    assert near_duplicate_groups([]) == []
    assert near_duplicate_groups(["안녕"]) == [0]


def test_delex_replaces_whole_tokens_with_particles(delex):
    assert delex("부산광역시의 기온") == f"{ENT_TOKEN}의 기온"
    assert delex("부산은 날씨 어때") == f"{ENT_TOKEN}은 날씨 어때"


def test_delex_skips_form_inside_longer_token(delex):
    assert delex("서울역 가는 길") == "서울역 가는 길"


def test_delex_without_match_returns_input(delex):
    assert delex("") == ""
    assert delex("오늘 날씨") == "오늘 날씨"


def test_delexicalizer_requires_forms(kiwi):
    with pytest.raises(BenchError):
        Delexicalizer([], kiwi)


def test_expand_and_delex_share_the_runtime_input_path(delex):
    text = "  부산은   날씨 어때 "
    assert prepare(Setup("expand", "char", "logreg"), text, delex) == "부산은 날씨 어때"
    assert prepare(Setup("delex", "char", "logreg"), text, delex) == f"{ENT_TOKEN}은 날씨 어때"


def test_delex_training_rows_use_runtime_delexicalizer(delex):
    texts, labels = training_rows(Setup("delex", "char", "logreg"), {"weather": ["{location} 날씨", "부산은 비 와?"]}, FORMS, delex)
    assert texts == [f"{ENT_TOKEN} 날씨", f"{ENT_TOKEN}은 비 와?"]
    assert labels == ["weather", "weather"]


def test_decide_rejects_below_threshold():
    classes = np.array(["a", OOS])
    probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    assert decide(probs, classes, 0.6) == ["a", OOS]
    assert decide(probs, classes, 0.0) == ["a", "a"]


def test_decide_requires_out_of_scope_class():
    with pytest.raises(BenchError):
        decide(np.array([[1.0]]), np.array(["a"]), 0.5)


def test_oos_recall_counts_rejections_and_needs_oos_rows():
    assert oos_recall(["a", OOS, OOS], ["a", OOS, "a"]) == pytest.approx(0.5)
    with pytest.raises(BenchError):
        oos_recall(["a"], ["a"])


def test_choose_threshold_meets_oos_target_when_possible():
    classes = np.array(["a", OOS])
    y_true = ["a", "a", OOS]
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4]])  # OOS row wrongly "a" at 0.6
    choice = choose_threshold(y_true, probs, classes, min_oos_recall=1.0)
    assert 0.6 < choice.threshold <= 0.8
    assert choice.constrained and choice.oos_recall == 1.0 and choice.macro_f1 == pytest.approx(1.0)


def test_choose_threshold_flags_unreachable_target():
    classes = np.array(["a", OOS])
    y_true = ["a", OOS]
    probs = np.array([[0.5, 0.5], [0.99, 0.01]])  # rejecting the OOS row needs t > 0.99; the grid stops at 0.95
    choice = choose_threshold(y_true, probs, classes, min_oos_recall=1.0)
    assert not choice.constrained


def test_choose_threshold_empty_raises():
    with pytest.raises(BenchError):
        choose_threshold([], np.zeros((0, 2)), np.array(["a", "b"]), 0.8)


def test_clarify_stats_counts_close_calls_only():
    classes = np.array(["a", "b", OOS])
    y_true = ["a", "b", OOS]
    probs = np.array([[0.50, 0.45, 0.05], [0.10, 0.85, 0.05], [0.5, 0.45, 0.05]])
    rate, top2 = clarify_stats(y_true, decide(probs, classes, 0.0), probs, classes)
    assert rate == pytest.approx(0.5) and top2 == pytest.approx(1.0)


TOY = {
    "weather": ["{location} 날씨 어때", "비 와?", "{location} 날씨 알려줘", "오늘 날씨", "내일 비 와?", "{location}에 눈 와?"],
    "greeting": ["안녕하세요", "반가워요", "안녕", "하이", "좋은 아침", "안녕하십니까"],
    OOS: ["주식 추천해줘", "노래 불러줘", "번역해줘", "요리 알려줘", "영화 추천", "농담 해줘"],
}


class FakeEmbedder:
    """Deterministic bag-of-characters embedding; no model download."""

    def __init__(self):
        self.fresh_calls = 0

    def encode_fresh(self, texts):
        self.fresh_calls += 1
        return self.encode(texts)

    def encode(self, texts):
        vecs = np.zeros((len(texts), 64), dtype=np.float32)
        for r, t in enumerate(texts):
            for ch in t:
                vecs[r, ord(ch) % 64] += 1
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


@pytest.mark.parametrize("clf", ["logreg", "svc"])
def test_ngram_model_learns_toy_data_and_is_deterministic(clf, kiwi, delex):
    setup = Setup("delex", "both", clf)
    texts, labels = training_rows(setup, TOY, FORMS, delex)
    a = IntentModel(setup, kiwi, None).fit(texts, labels)
    b = IntentModel(setup, kiwi, None).fit(texts, labels)
    probe = [prepare(setup, s, delex) for s in ["서울 날씨 어때요", "안녕하세요 반가워요"]]
    (pa, sa), (pb, _) = a.predict(probe), b.predict(probe)
    np.testing.assert_allclose(pa, pb)
    assert sa.shape == (2,) and np.all((sa >= 0) & (sa <= 1 + 1e-6))  # TF-IDF cosine gate
    assert list(a.classes_[pa.argmax(axis=1)]) == ["weather", "greeting"]


@pytest.mark.parametrize("features", ["emb:fake", "emb:fake+both"])
def test_embedding_model_reports_similarity_to_in_scope_rows_only(features, kiwi, delex):
    setup = Setup("expand", features, "logreg", 10.0)
    model = IntentModel(setup, kiwi, FakeEmbedder()).fit(*training_rows(setup, TOY, FORMS, delex))
    probs, sims = model.predict(["안녕하세요", "주식 추천해줘"])
    assert probs.shape == (2, 3)
    assert sims[0] == pytest.approx(1.0, abs=1e-5)  # exact in-scope training row
    assert sims[1] < 1.0  # OOS training rows are not part of the similarity index


def test_fresh_model_never_uses_cache(kiwi, delex):
    emb = FakeEmbedder()
    setup = Setup("expand", "emb:fake", "logreg")
    IntentModel(setup, kiwi, emb, fresh=True).fit(*training_rows(setup, TOY, FORMS, delex)).predict(["안녕"])
    assert emb.fresh_calls == 2


def test_embedder_mismatch_raises(kiwi):
    with pytest.raises(BenchError):
        IntentModel(Setup("expand", "emb:fake", "logreg"), kiwi, None)
    with pytest.raises(BenchError):
        IntentModel(Setup("expand", "char", "logreg"), kiwi, FakeEmbedder())


def test_setup_embedding_key_parsing():
    assert Setup("expand", "emb:arctic+both", "logreg").embedding == "arctic"
    assert Setup("expand", "emb:e5s", "logreg").embedding == "e5s"
    assert Setup("expand", "both", "logreg").embedding is None


def test_decide_similarity_gate_rejects_far_inputs():
    classes = np.array(["a", OOS])
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    assert decide(probs, classes, 0.5, np.array([0.8, 0.4]), 0.5) == ["a", OOS]


def test_choose_threshold_uses_similarity_when_probability_cannot_separate():
    classes = np.array(["a", OOS])
    y_true = ["a", "a", OOS]
    probs = np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01]])  # identical confidences
    sims = np.array([0.9, 0.85, 0.5])
    choice = choose_threshold(y_true, probs, classes, 1.0, sims)
    assert choice.constrained and 0.5 < choice.sim_threshold <= 0.85


def test_model_predict_on_empty_input_raises(kiwi):
    with pytest.raises(BenchError):
        IntentModel(Setup("expand", "char", "logreg"), kiwi, None).predict([])


def test_macro_f1_idx_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 200), rng.integers(0, 5, 200)
    assert macro_f1_idx(y, p, 5) == pytest.approx(f1_score(y, p, average="macro", zero_division=0))
    with pytest.raises(BenchError):
        macro_f1_idx(np.array([], dtype=int), np.array([], dtype=int), 3)


def test_semantic_groups_complete_link_does_not_chain():
    a, b, c = np.array([1.0, 0.0]), np.array([0.94, 0.34]), np.array([0.77, 0.64])  # a~b, b~c close; a~c far
    vecs = np.vstack([a, b / np.linalg.norm(b), c / np.linalg.norm(c)])
    groups = semantic_groups(vecs, threshold=0.9)
    assert len(set(groups)) == 2
    assert semantic_groups(np.zeros((0, 2))) == [] and semantic_groups(vecs[:1]) == [0]


def test_combine_groups_unions_relations_and_checks_lengths():
    assert combine_groups([0, 0, 1, 2], [0, 1, 1, 2]) == [0, 0, 0, 1]
    with pytest.raises(BenchError):
        combine_groups([0, 1], [0])


def test_template_groups_never_cross_intents():
    groups = template_groups({"a": ["날씨 알려줘", "날씨 알려줘요"], "b": ["날씨 알려줘"]})
    assert groups[0] == groups[1] != groups[2]


def test_split_halves_covers_every_group_per_intent():
    half = split_halves([0, 0, 1, 2, 3], ["a", "a", "a", "b", "b"])
    assert set(half) == {0, 1, 2, 3}
    assert {half[0], half[1]} == {0, 1} and {half[2], half[3]} == {0, 1}


def test_wilson_interval():
    lo, hi = wilson(89, 100)
    assert 0.80 < lo < 0.89 < hi < 0.95
    with pytest.raises(BenchError):
        wilson(0, 0)


def _toy_eval(kiwi, oos_eval, far_mask=None, features="emb:fake"):
    setup = Setup("expand", features, "logreg", 10.0)
    emb = FakeEmbedder() if setup.embedding else None
    return evaluate(setup, TOY, FORMS, oos_eval, kiwi, emb, far_mask=far_mask)


def test_evaluate_thresholds_do_not_depend_on_heldout_set(kiwi):
    a = _toy_eval(kiwi, ["주식 사도 돼?", "택배 조회"])
    b = _toy_eval(kiwi, ["완전히 다른 문장", "또 다른 문장", "세 번째"])
    assert a.choice == b.choice and a.f1_mean == b.f1_mean
    assert 0.0 <= a.oos_reject <= 1.0 and a.oos_reject_ci[0] <= a.oos_reject <= a.oos_reject_ci[1]


def test_evaluate_far_mask_must_match_heldout(kiwi):
    with pytest.raises(BenchError):
        _toy_eval(kiwi, ["주식 사도 돼?", "택배 조회"], far_mask=np.array([True]))


def test_paired_bootstrap_identical_results_give_zero(kiwi):
    r = _toy_eval(kiwi, ["주식 사도 돼?"], features="char")
    d, lo, hi = paired_bootstrap(r, r, n=50)
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_recommend_prefers_simpler_setup_when_indistinguishable(kiwi):
    import dataclasses

    base = _toy_eval(kiwi, ["주식 사도 돼?"], features="char")
    simple = dataclasses.replace(base, setup=Setup("expand", "emb:x", "logreg"), f1_mean=0.95, p95_ms=5.0, train_s=0.1, oos_reject=0.9)
    hybrid = dataclasses.replace(base, setup=Setup("expand", "emb:x+both", "logreg"), f1_mean=0.951, p95_ms=4.0, train_s=0.1, oos_reject=0.9)
    failing = dataclasses.replace(base, setup=Setup("expand", "char", "logreg"), f1_mean=0.99, p95_ms=1.0, train_s=0.1, oos_reject=0.1)
    assert recommend([hybrid, simple, failing], n=50) is simple  # same rows → difference 0 → tied
    assert recommend([failing], n=50) is None


def test_recommend_keeps_a_significantly_better_setup(kiwi):
    import dataclasses

    base = _toy_eval(kiwi, ["주식 사도 돼?"], features="char")
    rows = base.rows
    wrong = dict(rows, pred=(rows["y"] + 1) % rows["n_classes"])  # every prediction wrong
    good = dataclasses.replace(base, setup=Setup("expand", "emb:x+both", "logreg"), f1_mean=0.95, p95_ms=9.0, train_s=0.1, oos_reject=0.9,
                               rows=dict(rows, pred=rows["y"].copy()))
    bad = dataclasses.replace(base, setup=Setup("expand", "emb:x", "logreg"), f1_mean=0.91, p95_ms=1.0, train_s=0.1, oos_reject=0.9, rows=wrong)
    assert recommend([bad, good], n=200) is good  # simpler/faster but clearly worse → not tied
