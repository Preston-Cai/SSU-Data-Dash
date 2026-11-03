"""Mocked training/cache lifecycle checks; no GPU fits or neural downloads."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import data
import ensemble_runtime
import models
import semantic_text
from test_starter import make_listings


@pytest.fixture
def fake_members(tmp_path, monkeypatch):
    monkeypatch.setenv("SSU_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SSU_DISABLE_CACHE", raising=False)
    monkeypatch.setattr(ensemble_runtime, "base_signature", lambda: "mock-implementation-v1")
    calls = Counter()

    class FakeMember:
        def __init__(self, name):
            self.name = name

        def fit(self, train):
            calls[self.name] += 1
            self.mean_price = float(train.sold_price.mean())
            return self

        def predict(self, homes):
            scale = {"baseline": 1.0, "shallow": 0.98,
                     "deep": 1.02, "alternate_groups": 1.01}[self.name]
            return self.mean_price * scale + homes.bedrooms.to_numpy() * 100

    monkeypatch.setattr(models.DiverseEnsemble, "_make_member",
                        lambda self, name, params: FakeMember(name))
    return calls, FakeMember


def test_profile_selection_keeps_list_aware_uncorrected(monkeypatch):
    monkeypatch.delenv("MODEL_PROFILE", raising=False)
    from adaptive_comparables import AdaptiveComparableEnsemble, FixedSpecialistEnsemble
    from local_residual_model import LocalResidualEnsemble
    from comparable_model import ComparableEnsemble
    assert isinstance(models.make_model("off_market"), LocalResidualEnsemble)
    assert isinstance(models.make_model("list_aware"), FixedSpecialistEnsemble)
    monkeypatch.setenv("MODEL_PROFILE", "adaptive_comparables")
    assert isinstance(models.make_model("off_market"), AdaptiveComparableEnsemble)
    assert isinstance(models.make_model("list_aware"), FixedSpecialistEnsemble)
    monkeypatch.setenv("MODEL_PROFILE", "comparables")
    from comparable_model import ComparableEnsemble
    assert isinstance(models.make_model("off_market"), ComparableEnsemble)
    assert isinstance(models.make_model("list_aware"), ComparableEnsemble)
    monkeypatch.setenv("MODEL_PROFILE", "ensemble")
    off = models.make_model("off_market")
    assert isinstance(off, ensemble_runtime.CachedEnsemble)
    assert isinstance(models.make_model("list_aware"), ensemble_runtime.CachedEnsemble)
    monkeypatch.setenv("MODEL_PROFILE", "semantic")
    off = models.make_model("off_market")
    assert isinstance(off, models.TextCorrectedEnsemble) and off.semantic
    for profile in ["v11", "text_corrected"]:
        monkeypatch.setenv("MODEL_PROFILE", profile)
        off = models.make_model("off_market")
        assert isinstance(off, models.TextCorrectedEnsemble) and not off.semantic
        assert isinstance(models.make_model("list_aware"), ensemble_runtime.CachedEnsemble)


def test_cold_warm_cache_and_input_source_invalidation(fake_members, monkeypatch):
    calls, _ = fake_members
    train = make_listings(30)
    homes = data.hide_answers(make_listings(5, first_id=100), "off_market")
    cold = ensemble_runtime.CachedEnsemble("off_market").fit(train)
    expected = cold.predict(homes)
    assert not cold.cache_hit and sum(calls.values()) == 4
    warm = ensemble_runtime.CachedEnsemble("off_market").fit(train.copy())
    np.testing.assert_array_equal(warm.predict(homes.copy()), expected)
    assert warm.cache_hit and sum(calls.values()) == 4
    for name in cold.component_predictions:
        np.testing.assert_array_equal(warm.component_predictions[name], cold.component_predictions[name])

    changed_train = train.copy()
    changed_train.loc[0, "sold_price"] += 12345
    ensemble_runtime.CachedEnsemble("off_market").fit(changed_train).predict(homes)
    assert sum(calls.values()) == 8
    changed_homes = homes.copy()
    changed_homes.loc[0, "bedrooms"] += 1
    ensemble_runtime.CachedEnsemble("off_market").fit(train).predict(changed_homes)
    assert sum(calls.values()) == 12
    monkeypatch.setattr(ensemble_runtime, "base_signature", lambda: "mock-implementation-v2")
    ensemble_runtime.CachedEnsemble("off_market").fit(train).predict(homes)
    assert sum(calls.values()) == 16


def test_partial_member_cache_resumes_after_interruption(fake_members, monkeypatch):
    calls, FakeMember = fake_members
    train = make_listings(30)
    homes = data.hide_answers(make_listings(5, first_id=100), "off_market")

    def fail_at_deep(self, name, params):
        if name == "deep":
            raise RuntimeError("simulated interruption")
        return FakeMember(name)

    monkeypatch.setattr(models.DiverseEnsemble, "_make_member", fail_at_deep)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ensemble_runtime.CachedEnsemble("off_market").fit(train).predict(homes)
    assert calls == Counter({"baseline": 1, "shallow": 1})
    monkeypatch.setattr(models.DiverseEnsemble, "_make_member",
                        lambda self, name, params: FakeMember(name))
    result = ensemble_runtime.CachedEnsemble("off_market").fit(train).predict(homes)
    assert np.isfinite(result).all()
    assert calls == Counter({"baseline": 1, "shallow": 1, "deep": 1, "alternate_groups": 1})


def test_deferred_fit_owns_training_snapshot(fake_members):
    calls, _ = fake_members
    train = make_listings(30)
    original = train.copy(deep=True)
    homes = data.hide_answers(make_listings(5, first_id=100), "off_market")
    model = ensemble_runtime.CachedEnsemble("off_market").fit(train)
    train.loc[0, "sold_price"] *= 10
    prediction = model.predict(homes)
    expected = ensemble_runtime.CachedEnsemble("off_market").fit(original).predict(homes)
    np.testing.assert_array_equal(prediction, expected)
    assert sum(calls.values()) == 4
    assert model.train.sold_price.equals(original.sold_price)


def test_semantic_correction_uses_only_earlier_oof_rows(monkeypatch):
    train = make_listings(240)
    months = pd.period_range("2025-01", periods=8, freq="M")
    train["date_sold"] = [months[i // 30].start_time + pd.Timedelta(days=10)
                          for i in range(len(train))]
    train["date_listed"] = train.date_sold - pd.Timedelta(days=7)
    phrases = ["original kitchen needs updating", "renovated kitchen modern fixtures",
               "finished basement private entrance", "unfinished basement excellent potential",
               "corner apartment city views", "courtyard apartment new windows",
               "large garden detached garage", "compact balcony convenient transit"]
    train["remarks"] = [phrases[i % 8] for i in range(len(train))]
    sale_dates = train.set_index("ml_num").date_sold.to_dict()
    fitted = []
    query_ids = []

    class FakeEnsemble:
        def __init__(self, track):
            self.track = track

        def fit(self, past):
            self.train = past.copy()
            fitted.append(self.train)
            return self

        def predict(self, homes):
            assert "sold_price" not in homes and "date_sold" not in homes
            assert "list_price" not in homes
            assert not set(self.train.ml_num) & set(homes.ml_num)
            known_dates = [sale_dates[identifier] for identifier in homes.ml_num
                           if identifier in sale_dates]
            if known_dates:
                cutoff = min(known_dates).to_period("M").start_time
                assert self.train.date_sold.max() < cutoff
            query_ids.append(homes.ml_num.tolist())
            self.cache_hit = False
            result = 600000.0 + homes.bedrooms.to_numpy() * 1000
            self.component_predictions = {"baseline": result.copy()}
            return result

    embedding_inputs, released = [], []

    def fake_embeddings(texts):
        texts = list(texts)
        embedding_inputs.append(texts)
        return np.array([[len(text), text.count("kitchen"), text.count("original"), 1]
                         for text in texts], dtype=np.float32) / 100

    monkeypatch.setattr(ensemble_runtime, "CachedEnsemble", FakeEnsemble)
    monkeypatch.setattr(semantic_text, "check_dependencies", lambda: None)
    monkeypatch.setattr(semantic_text, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(semantic_text, "release_encoder", lambda: released.append(True))
    model = models.TextCorrectedEnsemble("off_market", semantic=True).fit(train)
    assert len(fitted) == 7  # six historical fits plus the full final fit
    assert len(embedding_inputs) == 1 and len(embedding_inputs[0]) == 180
    for audit in model.inner_fold_audit:
        assert pd.Timestamp(audit["train_max_sale"]) < pd.Period(audit["month"]).start_time
    pd.testing.assert_frame_equal(fitted[-1], train)
    before = model.semantic_center.copy()
    outer = data.hide_answers(make_listings(5, first_id=1000), "off_market")
    outer["remarks"] = "never observed in the historical correction"
    prediction = model.predict(outer)
    assert len(embedding_inputs) == 2 and len(embedding_inputs[1]) == len(outer)
    assert len(released) == 2
    np.testing.assert_array_equal(model.semantic_center, before)
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    assert "v11_prediction" in model.component_predictions
    np.testing.assert_allclose(
        prediction, model.component_predictions["uncorrected_ensemble"] * np.exp(model.text_log_adjustment))
