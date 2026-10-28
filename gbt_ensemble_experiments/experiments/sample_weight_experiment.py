"""Test time-decayed training weights on recency-encoded LightGBM."""

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

RECENT = runpy.run_path(str(HERE / "recency_encoding_experiment.py"), run_name="weight_recent")
TE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="weight_te")
LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="weight_lgb")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="weight_common")
WeightedTargetEncoder = RECENT["WeightedTargetEncoder"]
TargetEncoder = TE["TargetEncoder"]
FeatureBuilder = LGB["FeatureBuilder"]
target = LGB["target"]
invert = LGB["invert"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


def main():
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    train = listings[listings.date_sold < cutoff].copy()
    visible = data.hide_answers(valid, "off_market")

    builder = FeatureBuilder("off_market", use_remarks=True)
    base_train = builder.fit_transform(train)
    base_valid = builder.transform(visible)
    regular = TargetEncoder("off_market", 5.0)
    recent = WeightedTargetEncoder("off_market", cutoff, half_life_months=4.0, prior=5.0)
    x_train = pd.concat([base_train, regular.fit_transform(train), recent.fit_transform(train)], axis=1)
    x_valid = pd.concat([base_valid, regular.transform(visible), recent.transform(visible)], axis=1)

    rows = []
    output = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    age = (cutoff - train.date_sold).dt.days.to_numpy(float) / 30.44
    for half_life in (None, 12.0, 24.0, 48.0, 96.0):
        started = time.perf_counter()
        sample_weight = None if half_life is None else np.exp2(-np.maximum(age, 0) / half_life)
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=800,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            colsample_bytree=0.9,
            subsample=0.85,
            subsample_freq=1,
            reg_lambda=2.0,
            random_state=0,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            x_train,
            target(train, "off_market", residual=True),
            sample_weight=sample_weight,
            categorical_feature=list(builder.categories),
            eval_set=[(x_valid, target(valid, "off_market", residual=True))],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(70, verbose=False)],
        )
        prediction = invert(model.predict(x_valid, num_iteration=model.best_iteration_), visible, "off_market", residual=True)
        name = "none" if half_life is None else f"h{half_life:g}"
        output[f"off_market_weight_{name}"] = prediction
        result = {
            "training_weight_half_life": half_life,
            **metrics(valid.sold_price.to_numpy(float), prediction),
            "best_iteration": int(model.best_iteration_),
            "seconds": round(time.perf_counter() - started, 2),
        }
        rows.append(result)
        print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["mdape", "mape"])
    frame.to_csv(RESULTS / "sample_weight_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "sample_weight_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
