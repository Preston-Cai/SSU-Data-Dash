"""Leakage-controlled historical target encodings for LightGBM."""

from __future__ import annotations

import argparse
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

BASE = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="te_base")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="te_common")
FeatureBuilder = BASE["FeatureBuilder"]
target = BASE["target"]
invert = BASE["invert"]
metrics = COMMON["metrics"]
square_feet = COMMON["square_feet"]
RESULTS = HERE.parent / "results"
OFFICIAL_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")


def group_keys(rows: pd.DataFrame) -> dict[str, pd.Series]:
    community = rows.community.fillna("__MISSING__").astype(str)
    house_type = rows.house_type_name.fillna("__MISSING__").astype(str)
    style = rows.house_style.fillna("__MISSING__").astype(str)
    geo2 = rows.latitude.round(2).astype(str) + "|" + rows.longitude.round(2).astype(str)
    geo3 = rows.latitude.round(3).astype(str) + "|" + rows.longitude.round(3).astype(str)
    exact = rows.latitude.round(6).astype(str) + "|" + rows.longitude.round(6).astype(str)
    return {
        "community": community,
        "community_type": community + "|" + house_type,
        "community_style": community + "|" + style,
        "geo2_type": geo2 + "|" + house_type,
        "geo3_type": geo3 + "|" + house_type,
        "exact_type": exact + "|" + house_type,
    }


class TargetEncoder:
    """Smoothed leave-one-out encodings for fit rows, full-history for future rows."""

    def __init__(self, track: str, prior: float):
        self.track = track
        self.prior = prior
        self.tables = {}
        self.globals = {}

    def fit_transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        keys = group_keys(rows)
        sold = np.log(rows.sold_price.to_numpy(float))
        area = square_feet(rows.house_area).clip(lower=100).to_numpy(float)
        signals = {"log_price": sold, "log_ppsf": sold - np.log(area)}
        if self.track == "list_aware":
            signals["log_sale_list"] = np.log(rows.sold_price.to_numpy(float) / rows.list_price.to_numpy(float))
        encoded = pd.DataFrame(index=rows.index)
        for signal_name, values in signals.items():
            values_series = pd.Series(values, index=rows.index)
            global_value = float(values_series.mean())
            self.globals[signal_name] = global_value
            for group_name, key in keys.items():
                temp = pd.DataFrame({"key": key.to_numpy(), "value": values}, index=rows.index)
                sums = temp.groupby("key", observed=True).value.transform("sum")
                counts = temp.groupby("key", observed=True).value.transform("count")
                encoded[f"te_{signal_name}_{group_name}"] = (
                    sums - values_series + self.prior * global_value
                ) / (counts - 1 + self.prior)
                table = temp.groupby("key", observed=True).value.agg(["sum", "count"])
                self.tables[(signal_name, group_name)] = table
        return encoded

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        keys = group_keys(rows)
        encoded = pd.DataFrame(index=rows.index)
        for (signal_name, group_name), table in self.tables.items():
            key = keys[group_name]
            sums = key.map(table["sum"])
            counts = key.map(table["count"])
            encoded[f"te_{signal_name}_{group_name}"] = (
                sums.fillna(0.0) + self.prior * self.globals[signal_name]
            ) / (counts.fillna(0.0) + self.prior)
        return encoded


def screen():
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    rows = []
    output = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    for track in ("off_market", "list_aware"):
        months = None if track == "off_market" else 36
        train = listings[listings.date_sold < cutoff].copy()
        if months is not None:
            train = train[train.date_sold >= cutoff - pd.DateOffset(months=months)].copy()
        visible = data.hide_answers(valid, track)
        for prior in (5.0, 20.0, 50.0, 100.0):
            for objective in ("regression", "regression_l1"):
                started = time.perf_counter()
                builder = FeatureBuilder(track, use_remarks=True)
                x_train = builder.fit_transform(train)
                x_valid = builder.transform(visible)
                encoder = TargetEncoder(track, prior)
                te_train = encoder.fit_transform(train)
                te_valid = encoder.transform(visible)
                x_train = pd.concat([x_train, te_train], axis=1)
                x_valid = pd.concat([x_valid, te_valid], axis=1)
                model = lgb.LGBMRegressor(
                    objective=objective,
                    n_estimators=500,
                    learning_rate=0.05,
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
                    target(train, track, residual=True),
                    categorical_feature=list(builder.categories),
                    eval_set=[(x_valid, target(valid, track, residual=True))],
                    eval_metric="l1",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                raw = model.predict(x_valid, num_iteration=model.best_iteration_)
                predicted = invert(raw, visible, track, residual=True)
                score = metrics(valid.sold_price.to_numpy(float), predicted)
                name = f"{track}_te_p{prior:g}_{objective}"
                output[name] = predicted
                result = {
                    "track": track,
                    "prior": prior,
                    "objective": objective,
                    **score,
                    "best_iteration": int(model.best_iteration_),
                    "training_rows": len(train),
                    "seconds": round(time.perf_counter() - started, 2),
                }
                rows.append(result)
                print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    frame.to_csv(RESULTS / "target_encoding_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "target_encoding_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


def fit_fixed(train, homes, track, prior, objective, iterations):
    visible = data.hide_answers(homes, track)
    builder = FeatureBuilder(track, use_remarks=True)
    x_train = builder.fit_transform(train)
    x_homes = builder.transform(visible)
    encoder = TargetEncoder(track, prior)
    x_train = pd.concat([x_train, encoder.fit_transform(train)], axis=1)
    x_homes = pd.concat([x_homes, encoder.transform(visible)], axis=1)
    model = lgb.LGBMRegressor(
        objective=objective,
        n_estimators=iterations,
        learning_rate=0.05,
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
        target(train, track, residual=True),
        categorical_feature=list(builder.categories),
    )
    predicted = invert(model.predict(x_homes), visible, track, residual=True)
    return predicted


def walk_forward():
    listings = data.load("listings")
    # Choices fixed exclusively from November 2025 screening.
    configs = {
        "off_market": {
            "history_months": None,
            "prior": 5.0,
            "objective": "regression",
            "iterations": 405,
            "calibration": 0.967295,
        },
        "list_aware": {
            "history_months": 36,
            "prior": 100.0,
            "objective": "regression_l1",
            "iterations": 146,
            "calibration": 0.982791,
        },
    }
    frames = []
    for month in OFFICIAL_MONTHS:
        homes = listings[listings.date_sold.dt.to_period("M") == month].copy()
        for track, config in configs.items():
            train = listings[listings.date_sold < month.start_time].copy()
            if config["history_months"] is not None:
                train = train[
                    train.date_sold >= month.start_time - pd.DateOffset(months=config["history_months"])
                ].copy()
            started = time.perf_counter()
            raw = fit_fixed(
                train,
                homes,
                track,
                config["prior"],
                config["objective"],
                config["iterations"],
            )
            calibrated = raw * config["calibration"]
            print(
                month,
                track,
                "raw",
                json.dumps(metrics(homes.sold_price.to_numpy(float), raw)),
                "calibrated",
                json.dumps(metrics(homes.sold_price.to_numpy(float), calibrated)),
                "seconds",
                round(time.perf_counter() - started, 2),
                flush=True,
            )
            frames.append(pd.DataFrame({
                "ml_num": homes.ml_num.to_numpy(),
                "track": track,
                "month": str(month),
                "house_type_name": homes.house_type_name.to_numpy(),
                "sold_price": homes.sold_price.to_numpy(float),
                "predicted_price_raw": raw,
                "predicted_price": calibrated,
            }))
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(RESULTS / "target_encoding_walk_forward_predictions.csv", index=False)
    report = {}
    for track in ("off_market", "list_aware"):
        subset = predictions[predictions.track == track]
        report[track] = {
            "raw": metrics(subset.sold_price.to_numpy(float), subset.predicted_price_raw.to_numpy(float)),
            "calibrated": metrics(subset.sold_price.to_numpy(float), subset.predicted_price.to_numpy(float)),
        }
    (RESULTS / "target_encoding_walk_forward_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "walk-forward"), nargs="?", default="screen")
    {"screen": screen, "walk-forward": walk_forward}[parser.parse_args().command]()
