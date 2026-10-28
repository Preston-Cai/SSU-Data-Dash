"""Fast LightGBM experiments, isolated from all starter/custom project code."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="isolated_common")
make_raw_features = COMMON["make_features"]
metrics = COMMON["metrics"]
restrict_history_common = COMMON["restrict_history"]

TRACKS = ("off_market", "list_aware")
OFFICIAL_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")
RESULTS = HERE.parent / "results"
ARTIFACTS = HERE.parent / "artifacts"


@dataclass(frozen=True)
class LGBSpec:
    name: str
    objective: str = "regression"
    iterations: int = 300
    learning_rate: float = 0.07
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 80
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    l1: float = 0.0
    l2: float = 1.0
    history_months: int | None = 36
    residual_target: bool = True
    use_remarks: bool = True
    seed: int = 0


SPECS = (
    LGBSpec(name="lgb_l2_leaf31_h36"),
    LGBSpec(name="lgb_l1_leaf31_h36", objective="regression_l1"),
    LGBSpec(name="lgb_l2_leaf63_h36", num_leaves=63, min_child_samples=100),
    LGBSpec(name="lgb_l2_leaf31_h24", history_months=24),
    LGBSpec(name="lgb_l2_leaf31_all", history_months=None),
    LGBSpec(name="lgb_absolute_leaf31_h36", residual_target=False),
)


class FeatureBuilder:
    def __init__(self, track: str, use_remarks: bool):
        self.track = track
        self.use_remarks = use_remarks
        self.categories: dict[str, list[str]] = {}

    def fit_transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        frame, cats = make_raw_features(homes, self.track, self.use_remarks)
        self.categories = {
            column: sorted(frame[column].fillna("__MISSING__").astype(str).unique().tolist())
            for column in cats
        }
        return self._cast(frame)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        frame, _ = make_raw_features(homes, self.track, self.use_remarks)
        return self._cast(frame)

    def _cast(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column, values in self.categories.items():
            dtype = pd.CategoricalDtype(categories=values)
            frame[column] = frame[column].fillna("__MISSING__").astype(str).astype(dtype)
        return frame


def target(rows: pd.DataFrame, track: str, residual: bool) -> np.ndarray:
    sold = rows["sold_price"].to_numpy(float)
    if track == "list_aware" and residual:
        return np.log(sold / rows["list_price"].to_numpy(float))
    return np.log(sold)


def invert(raw: np.ndarray, homes: pd.DataFrame, track: str, residual: bool) -> np.ndarray:
    if track == "list_aware" and residual:
        return homes["list_price"].to_numpy(float) * np.exp(raw)
    return np.exp(raw)


def restrict_history(rows: pd.DataFrame, cutoff: pd.Timestamp, spec: LGBSpec) -> pd.DataFrame:
    earlier = rows[rows.date_sold < cutoff]
    if spec.history_months is not None:
        earlier = earlier[earlier.date_sold >= cutoff - pd.DateOffset(months=spec.history_months)]
    return earlier.copy()


def fit(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    track: str,
    spec: LGBSpec,
    *,
    early_stop: bool = True,
) -> tuple[lgb.LGBMRegressor, FeatureBuilder]:
    builder = FeatureBuilder(track, spec.use_remarks)
    x_train = builder.fit_transform(train)
    x_valid = builder.transform(data.hide_answers(valid, track))
    model = lgb.LGBMRegressor(
        objective=spec.objective,
        n_estimators=spec.iterations,
        learning_rate=spec.learning_rate,
        num_leaves=spec.num_leaves,
        max_depth=spec.max_depth,
        min_child_samples=spec.min_child_samples,
        colsample_bytree=spec.feature_fraction,
        subsample=spec.bagging_fraction,
        subsample_freq=1,
        reg_alpha=spec.l1,
        reg_lambda=spec.l2,
        random_state=spec.seed,
        n_jobs=-1,
        verbosity=-1,
        max_bin=255,
    )
    callbacks = [lgb.early_stopping(40, verbose=False)] if early_stop else []
    model.fit(
        x_train,
        target(train, track, spec.residual_target),
        categorical_feature=list(builder.categories),
        eval_set=[(x_valid, target(valid, track, spec.residual_target))],
        eval_metric="l1",
        callbacks=callbacks,
    )
    return model, builder


def predict(
    model: lgb.LGBMRegressor,
    builder: FeatureBuilder,
    homes: pd.DataFrame,
    track: str,
    spec: LGBSpec,
) -> np.ndarray:
    raw = model.predict(builder.transform(homes), num_iteration=model.best_iteration_)
    return invert(raw, homes, track, spec.residual_target)


def applicable_specs(track: str):
    for spec in SPECS:
        if track == "off_market" and spec.name == "lgb_absolute_leaf31_h36":
            continue
        yield spec


def screen() -> None:
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    rows = []
    prediction_frame = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    RESULTS.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        for spec in applicable_specs(track):
            train = restrict_history(listings, cutoff, spec)
            started = time.perf_counter()
            model, builder = fit(train, valid, track, spec)
            visible = data.hide_answers(valid, track)
            predicted = predict(model, builder, visible, track, spec)
            result = {
                "track": track,
                **asdict(spec),
                **metrics(valid.sold_price.to_numpy(float), predicted),
                "seconds": round(time.perf_counter() - started, 2),
                "best_iteration": int(model.best_iteration_),
                "training_rows": len(train),
            }
            rows.append(result)
            prediction_frame[f"{track}_{spec.name}"] = predicted
            print(json.dumps(result), flush=True)
    frame = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    frame.to_csv(RESULTS / "lightgbm_screen_november_2025.csv", index=False)
    prediction_frame.to_csv(RESULTS / "lightgbm_screen_predictions.csv", index=False)
    print(frame[["track", "name", "mape", "mdape", "median_signed_error", "best_iteration", "seconds"]].to_string(index=False))


def selected_specs() -> dict[str, LGBSpec]:
    path = RESULTS / "lightgbm_screen_november_2025.csv"
    if not path.exists():
        raise SystemExit("Run `screen` first.")
    frame = pd.read_csv(path).sort_values(["track", "mdape", "mape"])
    lookup = {spec.name: spec for spec in SPECS}
    chosen = {}
    for track in TRACKS:
        row = frame[frame.track == track].iloc[0]
        # Add a small margin because the final fit does not monitor the future
        # validation month used for early stopping.
        iterations = max(50, int(round(float(row.best_iteration) * 1.05)))
        chosen[track] = replace(lookup[row["name"]], iterations=iterations)
    return chosen


def walk_forward() -> None:
    listings = data.load("listings")
    specs = selected_specs()
    frames = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for month in OFFICIAL_MONTHS:
        homes = listings[listings.date_sold.dt.to_period("M") == month].copy()
        # Use the last part of the eligible training history only for harmless
        # iteration monitoring; it remains strictly earlier than the homes.
        for track in TRACKS:
            spec = specs[track]
            eligible = restrict_history(listings, month.start_time, spec)
            monitor_cutoff = eligible.date_sold.max() - pd.DateOffset(days=30)
            fit_rows = eligible[eligible.date_sold < monitor_cutoff].copy()
            monitor = eligible[eligible.date_sold >= monitor_cutoff].copy()
            model, builder = fit(fit_rows, monitor, track, spec, early_stop=False)
            visible = data.hide_answers(homes, track)
            predicted = predict(model, builder, visible, track, spec)
            score = metrics(homes.sold_price.to_numpy(float), predicted)
            print(month, track, json.dumps(score), flush=True)
            frames.append(pd.DataFrame({
                "ml_num": homes.ml_num.to_numpy(),
                "track": track,
                "month": str(month),
                "house_type_name": homes.house_type_name.to_numpy(),
                "sold_price": homes.sold_price.to_numpy(float),
                "predicted_price": predicted,
                "model": spec.name,
            }))
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(RESULTS / "lightgbm_walk_forward_predictions.csv", index=False)
    report = {
        track: metrics(
            predictions.loc[predictions.track == track, "sold_price"].to_numpy(float),
            predictions.loc[predictions.track == track, "predicted_price"].to_numpy(float),
        )
        for track in TRACKS
    }
    (RESULTS / "lightgbm_walk_forward_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "walk-forward"))
    return parser.parse_args()


if __name__ == "__main__":
    {"screen": screen, "walk-forward": walk_forward}[parse_args().command]()
