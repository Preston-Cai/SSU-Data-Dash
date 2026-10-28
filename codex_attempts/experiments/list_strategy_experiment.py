"""Explicit list-pricing strategy features for sale-to-list prediction."""

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

RECENT = runpy.run_path(str(HERE / "recency_encoding_experiment.py"), run_name="strategy_recent")
TE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="strategy_te")
LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="strategy_lgb")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="strategy_common")
WeightedTargetEncoder = RECENT["WeightedTargetEncoder"]
TargetEncoder = TE["TargetEncoder"]
FeatureBuilder = LGB["FeatureBuilder"]
target = LGB["target"]
invert = LGB["invert"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


def add_strategy_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    log_list = frame["log_list_price"]
    log_list_ppsf = np.log(frame["list_price_per_sqft"].clip(lower=1))
    for column in list(frame.columns):
        if "log_price_" in column and (column.startswith("te_") or column.startswith("recent")):
            frame[f"list_gap_{column}"] = log_list - frame[column]
        elif "log_ppsf_" in column and (column.startswith("te_") or column.startswith("recent")):
            frame[f"list_ppsf_gap_{column}"] = log_list_ppsf - frame[column]
    price = frame["list_price"].round().astype("int64")
    for unit in (1_000, 5_000, 10_000, 50_000, 100_000):
        remainder = price % unit
        frame[f"list_remainder_{unit}"] = remainder / unit
        frame[f"list_distance_round_{unit}"] = np.minimum(remainder, unit - remainder) / unit
    frame["list_ends_999"] = (price % 1_000 == 999).astype(int)
    frame["list_ends_900"] = (price % 1_000 == 900).astype(int)
    frame["list_ends_000"] = (price % 1_000 == 0).astype(int)
    return frame


def main():
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    train = listings[
        (listings.date_sold < cutoff)
        & (listings.date_sold >= cutoff - pd.DateOffset(months=36))
    ].copy()
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    visible = data.hide_answers(valid, "list_aware")
    builder = FeatureBuilder("list_aware", use_remarks=True)
    base_train = builder.fit_transform(train)
    base_valid = builder.transform(visible)
    regular = TargetEncoder("list_aware", 100.0)
    recent = WeightedTargetEncoder("list_aware", cutoff, 4.0, prior=5.0)
    x_train = add_strategy_features(pd.concat([
        base_train, regular.fit_transform(train), recent.fit_transform(train)
    ], axis=1))
    x_valid = add_strategy_features(pd.concat([
        base_valid, regular.transform(visible), recent.transform(visible)
    ], axis=1))
    rows = []
    output = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    for leaves in (15, 31, 63):
        for objective in ("regression", "regression_l1"):
            started = time.perf_counter()
            model = lgb.LGBMRegressor(
                objective=objective,
                n_estimators=650,
                learning_rate=0.04,
                num_leaves=leaves,
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
                target(train, "list_aware", residual=True),
                categorical_feature=list(builder.categories),
                eval_set=[(x_valid, target(valid, "list_aware", residual=True))],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            prediction = invert(model.predict(x_valid, num_iteration=model.best_iteration_), visible, "list_aware", residual=True)
            name = f"list_strategy_l{leaves}_{objective}"
            output[name] = prediction
            result = {
                "leaves": leaves,
                "objective": objective,
                **metrics(valid.sold_price.to_numpy(float), prediction),
                "best_iteration": int(model.best_iteration_),
                "seconds": round(time.perf_counter() - started, 2),
            }
            rows.append(result)
            print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["mdape", "mape"])
    frame.to_csv(RESULTS / "list_strategy_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "list_strategy_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
