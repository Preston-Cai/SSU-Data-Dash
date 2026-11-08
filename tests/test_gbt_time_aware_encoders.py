"""Regression checks for the handout's strictly-earlier-sales rule."""

from pathlib import Path
import runpy

import numpy as np
import pandas as pd


EXPERIMENTS = Path(__file__).parents[1] / "gbt_ensemble_experiments" / "experiments"
TARGETS = runpy.run_path(str(EXPERIMENTS / "target_encoding_experiment.py"), run_name="test_time_te")
RECENCY = runpy.run_path(str(EXPERIMENTS / "recency_encoding_experiment.py"), run_name="test_time_recent")
PREDICT = runpy.run_path(
    str(EXPERIMENTS.parent / "predict_artifact.py"), run_name="test_median_ratio_predict"
)


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "date_sold": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-02-01", "2025-03-01"]),
        "sold_price": [100.0, 200.0, 400.0, 800.0],
        "list_price": [100.0, 200.0, 400.0, 800.0],
        "house_area": ["1000"] * 4,
        "community": ["A"] * 4,
        "house_type_name": ["Detached"] * 4,
        "house_style": ["Two Storey"] * 4,
        "latitude": [44.0] * 4,
        "longitude": [-79.0] * 4,
    })


def assert_strict_history(encoded: pd.Series) -> None:
    assert np.isnan(encoded.iloc[0])
    # Both February sales can see January, but neither can see the other.
    assert np.allclose(encoded.iloc[1:3], np.log(100.0))
    assert encoded.iloc[3] < np.log(800.0)


def test_regular_encoder_uses_only_strictly_earlier_dates():
    encoded = TARGETS["TargetEncoder"]("off_market", 1.0).fit_transform(sample_rows())
    assert_strict_history(encoded["te_log_price_community"])


def test_recency_encoder_uses_only_strictly_earlier_dates():
    encoder = RECENCY["WeightedTargetEncoder"](
        "off_market", pd.Timestamp("2025-04-01"), half_life_months=4.0, prior=1.0
    )
    encoded = encoder.fit_transform(sample_rows())
    assert_strict_history(encoded["recent4_log_price_community"])


def test_median_ratio_component_uses_saved_history_only():
    component = {
        "kind": "median_ratio",
        "table": pd.DataFrame({"median": [np.log(0.9)], "count": [10]}, index=["A"]),
        "global_median": np.log(0.8),
        "prior": 10.0,
        "group_columns": ["community"],
    }
    predicted = PREDICT["median_ratio_prediction"](component, sample_rows().iloc[[0]])
    expected_ratio = np.exp((10 * np.log(0.9) + 10 * np.log(0.8)) / 20)
    assert np.allclose(predicted, [100.0 * expected_ratio])


def test_median_ratio_component_can_use_community_and_property_type():
    component = {
        "kind": "median_ratio",
        "table": pd.DataFrame(
            {"median": [np.log(0.95)], "count": [20]}, index=["A|Detached"]
        ),
        "global_median": np.log(0.8),
        "prior": 10.0,
        "group_columns": ["community", "house_type_name"],
    }
    predicted = PREDICT["median_ratio_prediction"](component, sample_rows().iloc[[0]])
    expected_ratio = np.exp((20 * np.log(0.95) + 10 * np.log(0.8)) / 30)
    assert np.allclose(predicted, [100.0 * expected_ratio])
