"""XGBoost histogram models on leakage-safe local encodings."""

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

RECENT = runpy.run_path(str(HERE / "recency_encoding_experiment.py"), run_name="xgb_recent")
TE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="xgb_te")
LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="xgb_lgb")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="xgb_common")
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
    rows = []
    output = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    for track in ("off_market", "list_aware"):
        train = listings[listings.date_sold < cutoff].copy()
        if track == "list_aware":
            train = train[train.date_sold >= cutoff - pd.DateOffset(months=36)].copy()
        visible = data.hide_answers(valid, track)
        builder = FeatureBuilder(track, use_remarks=True)
        x_train = builder.fit_transform(train)
        x_valid = builder.transform(visible)
        regular = TargetEncoder(track, 5.0 if track == "off_market" else 100.0)
        recent = WeightedTargetEncoder(track, cutoff, 4.0, prior=5.0)
        x_train = pd.concat([x_train, regular.fit_transform(train), recent.fit_transform(train)], axis=1)
        x_valid = pd.concat([x_valid, regular.transform(visible), recent.transform(visible)], axis=1)
        for depth in (4, 6, 8):
            for objective in ("reg:squarederror", "reg:absoluteerror"):
                started = time.perf_counter()
                model = XGBRegressor(
                    objective=objective,
                    n_estimators=1200,
                    learning_rate=0.035,
                    max_depth=depth,
                    min_child_weight=20,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=3.0,
                    tree_method="hist",
                    max_bin=256,
                    enable_categorical=True,
                    max_cat_to_onehot=20,
                    max_cat_threshold=64,
                    eval_metric="mae",
                    early_stopping_rounds=70,
                    random_state=0,
                    n_jobs=-1,
                )
                model.fit(
                    x_train,
                    target(train, track, residual=True),
                    eval_set=[(x_valid, target(valid, track, residual=True))],
                    verbose=False,
                )
                predicted = invert(model.predict(x_valid), visible, track, residual=True)
                name = f"{track}_xgb_d{depth}_{objective.split(':')[1]}"
                output[name] = predicted
                result = {
                    "track": track,
                    "depth": depth,
                    "objective": objective,
                    **metrics(valid.sold_price.to_numpy(float), predicted),
                    "best_iteration": int(model.best_iteration),
                    "seconds": round(time.perf_counter() - started, 2),
                }
                rows.append(result)
                print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    frame.to_csv(RESULTS / "xgboost_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "xgboost_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
