"""Predict corrections around leakage-safe local price anchors."""

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

RECENT = runpy.run_path(str(HERE / "recency_encoding_experiment.py"), run_name="anchor_recent")
TE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="anchor_te")
LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="anchor_lgb")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="anchor_common")
WeightedTargetEncoder = RECENT["WeightedTargetEncoder"]
TargetEncoder = TE["TargetEncoder"]
FeatureBuilder = LGB["FeatureBuilder"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


def main():
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    train = listings[listings.date_sold < cutoff].copy()
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    visible = data.hide_answers(valid, "off_market")

    builder = FeatureBuilder("off_market", use_remarks=True)
    base_train = builder.fit_transform(train)
    base_valid = builder.transform(visible)
    regular = TargetEncoder("off_market", 5.0)
    recent = WeightedTargetEncoder("off_market", cutoff, 4.0, prior=5.0)
    regular_train = regular.fit_transform(train)
    regular_valid = regular.transform(visible)
    recent_train = recent.fit_transform(train)
    recent_valid = recent.transform(visible)
    x_train = pd.concat([base_train, regular_train, recent_train], axis=1)
    x_valid = pd.concat([base_valid, regular_valid, recent_valid], axis=1)

    anchors = [
        "te_log_price_community",
        "te_log_price_community_type",
        "te_log_price_geo2_type",
        "recent4_log_price_community",
        "recent4_log_price_community_type",
        "recent4_log_price_geo2_type",
    ]
    rows = []
    output = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    log_sold = np.log(train.sold_price.to_numpy(float))
    for anchor in anchors:
        for objective in ("regression", "regression_l1"):
            started = time.perf_counter()
            model = lgb.LGBMRegressor(
                objective=objective,
                n_estimators=700,
                learning_rate=0.04,
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
            anchored_target = log_sold - x_train[anchor].to_numpy(float)
            valid_target = np.log(valid.sold_price.to_numpy(float)) - x_valid[anchor].to_numpy(float)
            model.fit(
                x_train,
                anchored_target,
                categorical_feature=list(builder.categories),
                eval_set=[(x_valid, valid_target)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            prediction = np.exp(x_valid[anchor].to_numpy(float) + model.predict(x_valid, num_iteration=model.best_iteration_))
            name = f"anchor_{anchor}_{objective}"
            output[name] = prediction
            result = {
                "anchor": anchor,
                "objective": objective,
                **metrics(valid.sold_price.to_numpy(float), prediction),
                "best_iteration": int(model.best_iteration_),
                "seconds": round(time.perf_counter() - started, 2),
            }
            rows.append(result)
            print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["mdape", "mape"])
    frame.to_csv(RESULTS / "anchor_residual_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "anchor_residual_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
