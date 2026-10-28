"""Official walk-forward predictions for the selected off-market XGBoost."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

RECENT = runpy.run_path(str(HERE / "recency_encoding_experiment.py"), run_name="xgbwf_recent")
TE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="xgbwf_te")
LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="xgbwf_lgb")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="xgbwf_common")
WeightedTargetEncoder = RECENT["WeightedTargetEncoder"]
TargetEncoder = TE["TargetEncoder"]
FeatureBuilder = LGB["FeatureBuilder"]
target = LGB["target"]
invert = LGB["invert"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


def fit_predict(listings, homes, cutoff):
    train = listings[listings.date_sold < cutoff].copy()
    visible = data.hide_answers(homes, "off_market")
    builder = FeatureBuilder("off_market", use_remarks=True)
    x_train = builder.fit_transform(train)
    x_homes = builder.transform(visible)
    regular = TargetEncoder("off_market", 5.0)
    recent = WeightedTargetEncoder("off_market", cutoff, 4.0, prior=5.0)
    x_train = pd.concat([x_train, regular.fit_transform(train), recent.fit_transform(train)], axis=1)
    x_homes = pd.concat([x_homes, regular.transform(visible), recent.transform(visible)], axis=1)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=868,
        learning_rate=0.035,
        max_depth=8,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        tree_method="hist",
        max_bin=256,
        enable_categorical=True,
        max_cat_to_onehot=20,
        max_cat_threshold=64,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x_train, target(train, "off_market", residual=True), verbose=False)
    return invert(model.predict(x_homes), visible, "off_market", residual=True)


def main():
    listings = data.load("listings")
    frames = []
    for month in pd.period_range("2025-12", "2026-05", freq="M"):
        homes = listings[listings.date_sold.dt.to_period("M") == month].copy()
        started = time.perf_counter()
        prediction = fit_predict(listings, homes, month.start_time)
        print(
            month,
            json.dumps(metrics(homes.sold_price.to_numpy(float), prediction)),
            "seconds",
            round(time.perf_counter() - started, 2),
            flush=True,
        )
        frames.append(pd.DataFrame({
            "ml_num": homes.ml_num.to_numpy(),
            "month": str(month),
            "sold_price": homes.sold_price.to_numpy(float),
            "xgboost_prediction": prediction,
        }))
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(RESULTS / "xgboost_walk_forward_predictions.csv", index=False)
    print(json.dumps(metrics(predictions.sold_price.to_numpy(float), predictions.xgboost_prediction.to_numpy(float)), indent=2))


if __name__ == "__main__":
    main()
