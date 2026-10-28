"""Exact official walk-forward for the November-frozen ensemble."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

TE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="ensemble_te")
RECENT = runpy.run_path(str(HERE / "recency_encoding_experiment.py"), run_name="ensemble_recent")
LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="ensemble_lgb")
TEXT = runpy.run_path(str(HERE / "linear_text_experiment.py"), run_name="ensemble_text")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="ensemble_common")

TargetEncoder = TE["TargetEncoder"]
WeightedTargetEncoder = RECENT["WeightedTargetEncoder"]
FeatureBuilder = LGB["FeatureBuilder"]
SparseFeatures = TEXT["SparseFeatures"]
target = LGB["target"]
invert = LGB["invert"]
text_target = TEXT["target"]
text_invert = TEXT["invert"]
metrics = COMMON["metrics"]

RESULTS = HERE.parent / "results"
MONTHS = pd.period_range("2025-12", "2026-05", freq="M")


CONFIGS = {
    "off_market": {
        "history_months": None,
        "regular_prior": 20.0,
        "regular_objective": "regression_l1",
        "regular_iterations": 522,
        "recent_regular_prior": 5.0,
        "recent_half_life": 24.0,
        "recent_objective": "regression",
        "recent_iterations": 628,
        "text_alpha": 1.0,
        "weights": (0.2, 0.6, 0.2),
        "calibration": 0.9659191220674491,
    },
    "list_aware": {
        "history_months": 36,
        "regular_prior": 5.0,
        "regular_objective": "regression_l1",
        "regular_iterations": 127,
        "recent_regular_prior": 100.0,
        "recent_half_life": 4.0,
        "recent_objective": "regression_l1",
        "recent_iterations": 198,
        "text_alpha": 50.0,
        "weights": (0.3, 0.5, 0.2),
        "calibration": 0.9834250678939119,
    },
}


def make_tree(track, config, x_train, y_train, builder, iterations, objective, learning_rate):
    model = lgb.LGBMRegressor(
        objective=objective,
        n_estimators=iterations,
        learning_rate=learning_rate,
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
    model.fit(x_train, y_train, categorical_feature=list(builder.categories))
    return model


def component_predictions(listings, homes, track, cutoff):
    config = CONFIGS[track]
    tree_train = listings[listings.date_sold < cutoff].copy()
    if config["history_months"] is not None:
        tree_train = tree_train[
            tree_train.date_sold >= cutoff - pd.DateOffset(months=config["history_months"])
        ].copy()
    visible = data.hide_answers(homes, track)

    builder = FeatureBuilder(track, use_remarks=True)
    base_train = builder.fit_transform(tree_train)
    base_homes = builder.transform(visible)

    regular = TargetEncoder(track, config["regular_prior"])
    regular_train = pd.concat([base_train, regular.fit_transform(tree_train)], axis=1)
    regular_homes = pd.concat([base_homes, regular.transform(visible)], axis=1)
    regular_model = make_tree(
        track,
        config,
        regular_train,
        target(tree_train, track, residual=True),
        builder,
        config["regular_iterations"],
        config["regular_objective"],
        0.05,
    )
    regular_prediction = invert(regular_model.predict(regular_homes), visible, track, residual=True)

    recent_regular = TargetEncoder(track, config["recent_regular_prior"])
    recent = WeightedTargetEncoder(track, cutoff, config["recent_half_life"], prior=5.0)
    recent_train = pd.concat([
        base_train,
        recent_regular.fit_transform(tree_train),
        recent.fit_transform(tree_train),
    ], axis=1)
    recent_homes = pd.concat([
        base_homes,
        recent_regular.transform(visible),
        recent.transform(visible),
    ], axis=1)
    recent_model = make_tree(
        track,
        config,
        recent_train,
        target(tree_train, track, residual=True),
        builder,
        config["recent_iterations"],
        config["recent_objective"],
        0.04,
    )
    recent_prediction = invert(recent_model.predict(recent_homes), visible, track, residual=True)

    text_train = listings[
        (listings.date_sold < cutoff)
        & (listings.date_sold >= cutoff - pd.DateOffset(months=36))
    ].copy()
    text_builder = SparseFeatures(track, "word", max_features=20_000, min_df=10)
    text_x_train = text_builder.fit_transform(text_train)
    text_x_homes = text_builder.transform(visible)
    text_model = Ridge(alpha=config["text_alpha"], solver="lsqr", tol=1e-4, max_iter=2000)
    text_model.fit(text_x_train, text_target(text_train, track))
    text_prediction = text_invert(text_model.predict(text_x_homes), visible, track)
    return regular_prediction, recent_prediction, text_prediction


def main():
    listings = data.load("listings")
    frames = []
    for month in MONTHS:
        homes = listings[listings.date_sold.dt.to_period("M") == month].copy()
        for track in ("off_market", "list_aware"):
            started = time.perf_counter()
            regular, recent, text = component_predictions(
                listings, homes, track, month.start_time
            )
            weights = CONFIGS[track]["weights"]
            prediction = np.exp(
                weights[0] * np.log(regular)
                + weights[1] * np.log(recent)
                + weights[2] * np.log(text)
            ) * CONFIGS[track]["calibration"]
            score = metrics(homes.sold_price.to_numpy(float), prediction)
            print(month, track, json.dumps(score), "seconds", round(time.perf_counter() - started, 2), flush=True)
            frames.append(pd.DataFrame({
                "ml_num": homes.ml_num.to_numpy(),
                "track": track,
                "month": str(month),
                "house_type_name": homes.house_type_name.to_numpy(),
                "sold_price": homes.sold_price.to_numpy(float),
                "regular_prediction": regular,
                "recent_prediction": recent,
                "text_prediction": text,
                "predicted_price": prediction,
            }))
    predictions = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(RESULTS / "ensemble_walk_forward_predictions.csv", index=False)
    report = {
        track: metrics(
            predictions.loc[predictions.track == track, "sold_price"].to_numpy(float),
            predictions.loc[predictions.track == track, "predicted_price"].to_numpy(float),
        )
        for track in ("off_market", "list_aware")
    }
    (RESULTS / "ensemble_walk_forward_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
