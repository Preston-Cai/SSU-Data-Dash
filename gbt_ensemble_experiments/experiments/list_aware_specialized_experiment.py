"""Prevalidation-only list-aware models with explicit local asking-price gaps."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="specialized_lgb")
TE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="specialized_te")
RECENT = runpy.run_path(str(HERE / "recency_encoding_experiment.py"), run_name="specialized_recent")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="specialized_common")
FeatureBuilder = LGB["FeatureBuilder"]
TargetEncoder = TE["TargetEncoder"]
WeightedTargetEncoder = RECENT["WeightedTargetEncoder"]
square_feet = COMMON["square_feet"]
RESULTS = HERE.parent / "results"


SPECS = {
    "gap_l1_leaf15": {"num_leaves": 15, "min_child_samples": 120, "n_estimators": 320},
    "gap_l1_leaf31": {"num_leaves": 31, "min_child_samples": 80, "n_estimators": 220},
    "gap_l1_leaf63": {"num_leaves": 63, "min_child_samples": 120, "n_estimators": 220},
}


def add_gap_features(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Expose list-price deviations from historical anchors directly."""
    frame = frame.copy()
    list_price = rows.list_price.to_numpy(float)
    area = square_feet(rows.house_area).clip(lower=100).to_numpy(float)
    log_list = np.log(list_price)
    log_list_ppsf = log_list - np.log(area)
    for column in list(frame.columns):
        if "_log_price_" in column:
            frame[f"gap_list_{column}"] = log_list - frame[column]
        elif "_log_ppsf_" in column:
            frame[f"gap_list_ppsf_{column}"] = log_list_ppsf - frame[column]
    # Asking-strategy features are known before sale and cheap to compute.
    frame["log_list_price_squared"] = log_list ** 2
    frame["list_price_mod_1000"] = np.mod(list_price, 1000.0) / 1000.0
    frame["list_price_mod_10000"] = np.mod(list_price, 10000.0) / 10000.0
    return frame


def matrices(train: pd.DataFrame, homes: pd.DataFrame, cutoff: pd.Timestamp):
    visible = data.hide_answers(homes, "list_aware")
    builder = FeatureBuilder("list_aware", use_remarks=True)
    x_train = builder.fit_transform(train)
    x_homes = builder.transform(visible)
    regular = TargetEncoder("list_aware", prior=100.0)
    recent = WeightedTargetEncoder("list_aware", cutoff, half_life_months=4.0, prior=5.0)
    x_train = pd.concat(
        [x_train, regular.fit_transform(train), recent.fit_transform(train)], axis=1
    )
    x_homes = pd.concat(
        [x_homes, regular.transform(visible), recent.transform(visible)], axis=1
    )
    return add_gap_features(x_train, train), add_gap_features(x_homes, visible), builder


def main() -> None:
    listings = data.load("listings")
    outputs = []
    scores = []
    for month in pd.period_range("2025-09", "2025-11", freq="M"):
        cutoff = month.start_time
        train = listings[
            (listings.date_sold < cutoff)
            & (listings.date_sold >= cutoff - pd.DateOffset(months=36))
        ].copy()
        homes = listings[listings.date_sold.dt.to_period("M") == month].copy()
        started = time.perf_counter()
        x_train, x_homes, builder = matrices(train, homes, cutoff)
        target = np.log(train.sold_price.to_numpy(float) / train.list_price.to_numpy(float))
        result = pd.DataFrame({
            "ml_num": homes.ml_num.to_numpy(),
            "month": str(month),
            "sold_price": homes.sold_price.to_numpy(float),
        })
        for name, spec in SPECS.items():
            model = lgb.LGBMRegressor(
                objective="regression_l1",
                learning_rate=0.04,
                reg_lambda=2.0,
                colsample_bytree=0.9,
                subsample=0.85,
                subsample_freq=1,
                random_state=0,
                n_jobs=-1,
                verbosity=-1,
                **spec,
            )
            model.fit(x_train, target, categorical_feature=list(builder.categories))
            predicted = homes.list_price.to_numpy(float) * np.exp(model.predict(x_homes))
            result[name] = predicted
            ape = 100 * np.abs(predicted - homes.sold_price.to_numpy(float)) / homes.sold_price.to_numpy(float)
            scores.append({
                "month": str(month),
                "model": name,
                "mape": float(np.mean(ape)),
                "mdape": float(np.median(ape)),
            })
        outputs.append(result)
        print(month, json.dumps(scores[-len(SPECS):]), "seconds", round(time.perf_counter() - started, 2), flush=True)
    predictions = pd.concat(outputs, ignore_index=True)
    predictions.to_csv(RESULTS / "list_aware_specialized_prevalidation_predictions.csv", index=False)
    pd.DataFrame(scores).to_csv(RESULTS / "list_aware_specialized_prevalidation_scores.csv", index=False)
    for name in SPECS:
        actual = predictions.sold_price.to_numpy(float)
        predicted = predictions[name].to_numpy(float)
        ape = 100 * np.abs(predicted - actual) / actual
        print(name, {"mape": float(np.mean(ape)), "mdape": float(np.median(ape))})


if __name__ == "__main__":
    main()
