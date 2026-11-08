"""Add recency-weighted leakage-safe local price encodings."""

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

BASE = runpy.run_path(str(HERE / "target_encoding_experiment.py"), run_name="recent_base")
LGB = runpy.run_path(str(HERE / "lightgbm_experiment.py"), run_name="recent_lgb")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="recent_common")
FeatureBuilder = LGB["FeatureBuilder"]
TargetEncoder = BASE["TargetEncoder"]
group_keys = BASE["group_keys"]
target = LGB["target"]
invert = LGB["invert"]
metrics = COMMON["metrics"]
square_feet = COMMON["square_feet"]
RESULTS = HERE.parent / "results"


class WeightedTargetEncoder:
    def __init__(self, track: str, cutoff: pd.Timestamp, half_life_months: float, prior: float = 5.0):
        self.track = track
        self.cutoff = cutoff
        self.half_life = half_life_months
        self.prior = prior
        self.tables = {}
        self.globals = {}

    def fit_transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.date_sold.isna().any():
            raise ValueError("Target encoding requires date_sold for every fit row")
        keys = group_keys(rows)
        age_months = (self.cutoff - rows.date_sold).dt.days.to_numpy(float) / 30.44
        weights = np.exp2(-np.maximum(age_months, 0) / self.half_life)
        sold = np.log(rows.sold_price.to_numpy(float))
        area = square_feet(rows.house_area).clip(lower=100).to_numpy(float)
        signals = {"log_price": sold, "log_ppsf": sold - np.log(area)}
        if self.track == "list_aware":
            signals["log_sale_list"] = np.log(rows.sold_price.to_numpy(float) / rows.list_price.to_numpy(float))
        encoded = pd.DataFrame(index=rows.index)
        for signal_name, values in signals.items():
            self.globals[signal_name] = float(np.average(values, weights=weights))

            dated_values = pd.DataFrame({
                "date_sold": rows.date_sold.to_numpy(),
                "weighted_value": weights * values,
                "weight": weights,
            })
            daily_global = dated_values.groupby("date_sold", sort=True)[
                ["weighted_value", "weight"]
            ].sum()
            daily_global[["past_weighted_value", "past_weight"]] = daily_global[
                ["weighted_value", "weight"]
            ].cumsum().shift()
            past_global_sum = rows.date_sold.map(daily_global.past_weighted_value)
            past_global_weight = rows.date_sold.map(daily_global.past_weight)
            past_global = past_global_sum / past_global_weight
            # `weights` are relative to the prediction cutoff.  Scaling the
            # prior by the current row's weight makes all historical weights
            # equivalent to ages measured from that row's sale date.
            prior_weight = self.prior * weights

            for group_name, key in keys.items():
                temp = pd.DataFrame({
                    "key": key.to_numpy(),
                    "weighted_value": weights * values,
                    "weight": weights,
                }, index=rows.index)
                dated = temp.assign(date_sold=rows.date_sold.to_numpy())
                daily = dated.groupby(["key", "date_sold"], observed=True)[
                    ["weighted_value", "weight"]
                ].sum()
                cumulative = daily.groupby(level="key", observed=True)[
                    ["weighted_value", "weight"]
                ].cumsum().groupby(level="key", observed=True).shift()
                daily[["past_weighted_value", "past_weight"]] = cumulative
                lookup = pd.MultiIndex.from_arrays([key.to_numpy(), rows.date_sold.to_numpy()])
                group_weighted_sum = pd.Series(
                    daily.past_weighted_value.reindex(lookup).to_numpy(), index=rows.index
                ).fillna(0.0)
                group_weight = pd.Series(
                    daily.past_weight.reindex(lookup).to_numpy(), index=rows.index
                ).fillna(0.0)
                encoded[f"recent{self.half_life:g}_{signal_name}_{group_name}"] = (
                    group_weighted_sum + prior_weight * past_global
                ) / (group_weight + prior_weight)
                self.tables[(signal_name, group_name)] = temp.groupby("key", observed=True)[
                    ["weighted_value", "weight"]
                ].sum()
        return encoded

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        keys = group_keys(rows)
        encoded = pd.DataFrame(index=rows.index)
        for (signal_name, group_name), table in self.tables.items():
            key = keys[group_name]
            weighted_sum = key.map(table.weighted_value)
            weight = key.map(table.weight)
            encoded[f"recent{self.half_life:g}_{signal_name}_{group_name}"] = (
                weighted_sum.fillna(0.0) + self.prior * self.globals[signal_name]
            ) / (weight.fillna(0.0) + self.prior)
        return encoded


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
        for half_life in (4.0, 8.0, 12.0, 24.0):
            started = time.perf_counter()
            builder = FeatureBuilder(track, use_remarks=True)
            x_train = builder.fit_transform(train)
            x_valid = builder.transform(visible)
            regular_prior = 5.0 if track == "off_market" else 100.0
            regular = TargetEncoder(track, regular_prior)
            recent = WeightedTargetEncoder(track, cutoff, half_life, prior=5.0)
            x_train = pd.concat([
                x_train,
                regular.fit_transform(train),
                recent.fit_transform(train),
            ], axis=1)
            x_valid = pd.concat([
                x_valid,
                regular.transform(visible),
                recent.transform(visible),
            ], axis=1)
            objective = "regression" if track == "off_market" else "regression_l1"
            model = lgb.LGBMRegressor(
                objective=objective,
                n_estimators=650,
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
            model.fit(
                x_train,
                target(train, track, residual=True),
                categorical_feature=list(builder.categories),
                eval_set=[(x_valid, target(valid, track, residual=True))],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            predicted = invert(model.predict(x_valid, num_iteration=model.best_iteration_), visible, track, residual=True)
            result = {
                "track": track,
                "half_life_months": half_life,
                **metrics(valid.sold_price.to_numpy(float), predicted),
                "best_iteration": int(model.best_iteration_),
                "seconds": round(time.perf_counter() - started, 2),
            }
            rows.append(result)
            output[f"{track}_recent_hl{half_life:g}"] = predicted
            print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    frame.to_csv(RESULTS / "recency_encoding_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "recency_encoding_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
