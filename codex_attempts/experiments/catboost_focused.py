"""Focused long-running CatBoost candidates with ordered categorical statistics."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="cat_focused_common")
make_features = COMMON["make_features"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


def target(rows, track):
    if track == "list_aware":
        return np.log(rows.sold_price.to_numpy(float) / rows.list_price.to_numpy(float))
    return np.log(rows.sold_price.to_numpy(float))


def invert(raw, homes, track):
    if track == "list_aware":
        return homes.list_price.to_numpy(float) * np.exp(raw)
    return np.exp(raw)


def main():
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    output = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    rows = []
    for track in ("off_market", "list_aware"):
        train = listings[listings.date_sold < cutoff].copy()
        if track == "list_aware":
            train = train[train.date_sold >= cutoff - pd.DateOffset(months=36)].copy()
        visible = data.hide_answers(valid, track)
        x_train, cats = make_features(train, track, use_remarks=True)
        x_valid, _ = make_features(visible, track, use_remarks=True)
        started = time.perf_counter()
        model = CatBoostRegressor(
            iterations=500,
            depth=7,
            learning_rate=0.07,
            loss_function="RMSE",
            eval_metric="MAE",
            l2_leaf_reg=6.0,
            random_strength=0.3,
            random_seed=0,
            boosting_type="Plain",
            bootstrap_type="Bernoulli",
            subsample=0.85,
            rsm=0.85,
            one_hot_max_size=20,
            max_ctr_complexity=1,
            allow_writing_files=False,
            thread_count=-1,
            verbose=50,
        )
        model.fit(
            x_train,
            target(train, track),
            cat_features=cats,
            eval_set=(x_valid, target(valid, track)),
            early_stopping_rounds=60,
            verbose=50,
        )
        prediction = invert(model.predict(x_valid), visible, track)
        output[f"{track}_catboost"] = prediction
        result = {
            "track": track,
            **metrics(valid.sold_price.to_numpy(float), prediction),
            "tree_count": model.tree_count_,
            "seconds": round(time.perf_counter() - started, 2),
        }
        rows.append(result)
        print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "catboost_focused_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "catboost_focused_screen_predictions.csv", index=False)


if __name__ == "__main__":
    main()
