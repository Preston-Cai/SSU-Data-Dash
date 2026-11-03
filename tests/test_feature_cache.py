"""Exact feature equivalence checks; no CatBoost fitting or downloads."""
import numpy as np
import pandas as pd
import pytest

import data
import feature_cache
import features
import models
from test_starter import make_listings


@pytest.fixture(autouse=True)
def fresh_feature_cache(monkeypatch):
    monkeypatch.delenv("SSU_DISABLE_CACHE", raising=False)
    feature_cache.clear()
    yield
    feature_cache.clear()


def test_history_reuse_is_exact_and_tracks_outcomes(monkeypatch):
    train = make_listings(240)
    homes = data.hide_answers(make_listings(20, seed=5, first_id=500), "off_market")
    calls = []
    original = models.FeatureModel._add_history_features

    def count_call(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(models.FeatureModel, "_add_history_features", count_call)
    first = models.CatBoostFeatureModel(features.off_market_features)
    first._build_history(train)
    expected = first._history_features(homes)
    second = models.CatBoostFeatureModel(features.off_market_features, depth=6)
    second._build_history(train.copy())
    actual = second._history_features(homes.copy())
    assert len(calls) == 1
    assert first.history is second.history
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    # Returning a table never gives callers a mutable reference to the cache.
    actual.iloc[0, 0] = -12345
    pd.testing.assert_frame_equal(second._history_features(homes), expected, check_exact=True)

    monkeypatch.setenv("SSU_DISABLE_CACHE", "1")
    uncached = models.CatBoostFeatureModel(features.off_market_features)
    uncached._build_history(train)
    pd.testing.assert_frame_equal(uncached._history_features(homes), expected, check_exact=True)
    assert len(calls) == 2

    monkeypatch.delenv("SSU_DISABLE_CACHE")
    changed = train.copy()
    changed.loc[changed.index[0], "sold_price"] *= 2
    second._build_history(changed)
    second._history_features(homes)
    assert len(calls) == 3
    assert second.history is not first.history


def test_svd_reuses_exact_fit_but_separates_component_counts(monkeypatch):
    train = make_listings(160)
    phrases = [
        "original kitchen needs updating older fixtures",
        "renovated kitchen modern fixtures polished floors",
        "finished basement private entrance rental suite",
        "unfinished basement excellent potential spacious rooms",
        "corner apartment expansive city skyline views",
        "quiet courtyard interior apartment new windows",
        "large garden detached garage family neighbourhood",
        "compact balcony convenient transit vibrant community",
    ]
    train["remarks"] = [phrases[i % len(phrases)] for i in range(len(train))]
    X = features.off_market_features(train)
    first = models.CatBoostFeatureModel(features.off_market_features, text_components=4)
    expected = first._add_text_features(X, train, fit=True)
    same = models.CatBoostFeatureModel(features.off_market_features, text_components=4, depth=6)
    actual = same._add_text_features(X, train.copy(), fit=True)
    assert first.text_vectorizer is same.text_vectorizer
    assert first.text_svd is same.text_svd
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    smaller = models.CatBoostFeatureModel(features.off_market_features, text_components=2)
    small_result = smaller._add_text_features(X, train, fit=True)
    assert smaller.text_svd is not first.text_svd
    assert len([column for column in small_result if column.startswith("remarks_component_")]) == 2

    monkeypatch.setenv("SSU_DISABLE_CACHE", "1")
    uncached = models.CatBoostFeatureModel(features.off_market_features, text_components=4)
    pd.testing.assert_frame_equal(uncached._add_text_features(X, train, fit=True), expected, check_exact=True)
    pd.testing.assert_frame_equal(uncached._add_text_features(X, train),
                                  same._add_text_features(X, train), check_exact=True)


def test_feature_cache_is_bounded():
    for index in range(20):
        feature_cache.put("text_fit", index, np.array([index]))
    assert len(feature_cache._CACHES["text_fit"]) == 4
    assert feature_cache.get("text_fit", 0) is None
