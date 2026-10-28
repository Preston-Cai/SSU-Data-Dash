"""Leakage-safe, isolated experiments for the SSU Data Dash challenge."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402


TRACKS = ("off_market", "list_aware")
OFFICIAL_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")
ATTEMPT_ROOT = HERE.parent
ARTIFACTS = ATTEMPT_ROOT / "artifacts"
RESULTS = ATTEMPT_ROOT / "results"

# Columns that are not available at inference, explicitly repeated here so a
# future edit to the starter cannot silently introduce leakage.
FORBIDDEN = {
    "sold_price",
    "date_sold",
    "days_on_market",
    "external_avm_estimate",
    "external_avm_estimate_date",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    iterations: int = 200
    depth: int = 8
    learning_rate: float = 0.05
    l2_leaf_reg: float = 5.0
    random_strength: float = 0.5
    loss_function: str = "RMSE"
    residual_target: bool = True
    use_remarks: bool = True
    seed: int = 0
    history_months: int | None = 36


SPECS = (
    ModelSpec(name="cat_resid_d6_h24", depth=6, history_months=24),
    ModelSpec(name="cat_resid_d7_h36", depth=7),
    ModelSpec(name="cat_resid_d7_all", depth=7, history_months=None),
    ModelSpec(
        name="cat_absolute_d7_h36",
        depth=7,
        residual_target=False,
    ),
)


CONDITION_PATTERNS = {
    "remark_renovated": r"renovat|fully updated|completely redone|top to bottom|turnkey|move.in ready",
    "remark_luxury": r"luxur|designer|premium|high.end|custom|executive",
    "remark_needs_work": r"as.is|needs? (?:work|attention|updat)|fixer|dated|original finish|older appliance|sweat equity",
    "remark_view": r"unobstructed view|lake view|city view|skyline|ravine|waterfront",
    "remark_transit": r"subway|transit|downtown commute|highway access",
    "remark_school": r"school|university|college",
    "remark_investor": r"investor|investment|rental|income potential",
    "remark_motivated": r"motivated|open to offers|priced to sell",
    "remark_new": r"brand.new|newly built|new construction",
    "remark_pool": r"\bpool\b",
}


def square_feet(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.extract(r"^\s*([\d.]+)(?:\s*-\s*([\d.]+))?")
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2.0


def make_features(homes: pd.DataFrame, track: str, use_remarks: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Create only fields available for the matching hidden-test track."""
    illegal = FORBIDDEN.intersection(homes.columns)
    # Training tables naturally contain forbidden columns, but none may be
    # copied below. This check protects inference frames instead of rejecting
    # the labeled table needed to fit a target.
    if track == "off_market" and "list_price" in homes.columns:
        available = homes.drop(columns=["list_price"])
    else:
        available = homes

    x = pd.DataFrame(index=homes.index)
    categorical = []
    for column in ("house_type_name", "house_style", "municipality", "community"):
        x[column] = available[column].fillna("__MISSING__").astype(str)
        categorical.append(column)

    area = square_feet(available["house_area"]).clip(lower=100)
    x["square_feet"] = area
    x["log_square_feet"] = np.log(area)
    for column in ("bedrooms", "washrooms", "parking_total", "garage", "lot_front", "lot_depth", "latitude", "longitude"):
        x[column] = pd.to_numeric(available[column], errors="coerce")
    x["lot_area"] = x["lot_front"] * x["lot_depth"]
    x["log_lot_area"] = np.log1p(x["lot_area"])
    x["bath_per_bed"] = x["washrooms"] / x["bedrooms"].clip(lower=1)
    x["area_per_bed"] = area / x["bedrooms"].clip(lower=1)
    x["area_per_bath"] = area / x["washrooms"].clip(lower=1)
    x["is_condo"] = available["house_type_name"].astype(str).str.contains("Condo|Co-Op", regex=True).astype(int)

    listed = pd.to_datetime(available["date_listed"])
    x["month_index"] = (listed.dt.year - 2021) * 12 + listed.dt.month - 1
    x["calendar_month_sin"] = np.sin(2 * np.pi * listed.dt.month / 12)
    x["calendar_month_cos"] = np.cos(2 * np.pi * listed.dt.month / 12)
    x["listed_day_of_year"] = listed.dt.dayofyear

    # Multi-resolution spatial cells let trees model very local effects while
    # retaining community-level fallbacks. They are derived from synthetic
    # coordinates only; no external map data is used.
    # Finer grids (3-4 decimals) create tens of thousands of categories and are
    # both slower and less stable out of time. Fine location stays numeric.
    for decimals in (2,):
        column = f"geo_{decimals}"
        x[column] = (
            available["latitude"].round(decimals).astype(str)
            + "_"
            + available["longitude"].round(decimals).astype(str)
        )
        categorical.append(column)

    interactions = {
        "community_type": available["community"].astype(str) + "|" + available["house_type_name"].astype(str),
        "community_style": available["community"].astype(str) + "|" + available["house_style"].astype(str),
        "municipality_type": available["municipality"].astype(str) + "|" + available["house_type_name"].astype(str),
        "type_style": available["house_type_name"].astype(str) + "|" + available["house_style"].astype(str),
    }
    for column, values in interactions.items():
        x[column] = values.fillna("__MISSING__")
        categorical.append(column)

    if track == "list_aware":
        price = pd.to_numeric(available["list_price"], errors="coerce").clip(lower=1)
        x["list_price"] = price
        x["log_list_price"] = np.log(price)
        x["list_price_per_sqft"] = price / area

    if use_remarks:
        remarks = available["remarks"].fillna("").astype(str).str.lower()
        x["remark_chars"] = remarks.str.len()
        x["remark_words"] = remarks.str.split().str.len()
        x["remark_exclamations"] = remarks.str.count("!")
        x["remark_dollar_values"] = remarks.str.count(r"\$[\d,]+")
        for column, pattern in CONDITION_PATTERNS.items():
            x[column] = remarks.str.contains(pattern, regex=True).astype(int)

    assert not FORBIDDEN.intersection(x.columns)
    assert not (track == "off_market" and "list_price" in x.columns)
    return x, categorical


def targets(train: pd.DataFrame, track: str, residual: bool) -> np.ndarray:
    sold = train["sold_price"].to_numpy(float)
    if track == "list_aware" and residual:
        return np.log(sold / train["list_price"].to_numpy(float))
    return np.log(sold)


def restrict_history(train: pd.DataFrame, cutoff: pd.Timestamp, spec: ModelSpec) -> pd.DataFrame:
    """Apply a trailing training window without ever crossing the cutoff."""
    earlier = train[train["date_sold"] < cutoff]
    if spec.history_months is not None:
        earlier = earlier[earlier["date_sold"] >= cutoff - pd.DateOffset(months=spec.history_months)]
    return earlier.copy()


def inverse_target(raw: np.ndarray, homes: pd.DataFrame, track: str, residual: bool) -> np.ndarray:
    if track == "list_aware" and residual:
        return homes["list_price"].to_numpy(float) * np.exp(raw)
    return np.exp(raw)


def fit_model(train: pd.DataFrame, track: str, spec: ModelSpec, eval_rows: pd.DataFrame | None = None) -> CatBoostRegressor:
    x_train, cats = make_features(train, track, spec.use_remarks)
    eval_set = None
    if eval_rows is not None and len(eval_rows):
        x_eval, _ = make_features(eval_rows, track, spec.use_remarks)
        eval_set = (x_eval, targets(eval_rows, track, spec.residual_target))
    model = CatBoostRegressor(
        iterations=spec.iterations,
        depth=spec.depth,
        learning_rate=spec.learning_rate,
        l2_leaf_reg=spec.l2_leaf_reg,
        random_strength=spec.random_strength,
        loss_function=spec.loss_function,
        eval_metric="MAPE",
        random_seed=spec.seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
        one_hot_max_size=20,
        max_ctr_complexity=1,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        rsm=0.9,
    )
    model.fit(
        x_train,
        targets(train, track, spec.residual_target),
        cat_features=cats,
        eval_set=eval_set,
        early_stopping_rounds=100 if eval_set is not None else None,
        verbose=False,
    )
    return model


def predict_model(model: CatBoostRegressor, homes: pd.DataFrame, track: str, spec: ModelSpec) -> np.ndarray:
    x, _ = make_features(homes, track, spec.use_remarks)
    raw = model.predict(x)
    return inverse_target(raw, homes, track, spec.residual_target)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    signed = 100 * (predicted - actual) / actual
    absolute = np.abs(signed)
    return {
        "n": int(len(actual)),
        "mape": round(float(np.mean(absolute)), 4),
        "mdape": round(float(np.median(absolute)), 4),
        "median_signed_error": round(float(np.median(signed)), 4),
    }


def screen() -> None:
    """Tune on November 2025, before the official validation period."""
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    end = pd.Timestamp("2025-12-01")
    train = listings[listings["date_sold"] < cutoff].copy()
    valid = listings[(listings["date_sold"] >= cutoff) & (listings["date_sold"] < end)].copy()
    rows = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        for spec in SPECS:
            if track == "off_market" and spec.name == "cat_absolute_d7_h36":
                continue
            started = time.perf_counter()
            spec_train = restrict_history(train, cutoff, spec)
            model = fit_model(spec_train, track, spec)
            predicted = predict_model(model, data.hide_answers(valid, track), track, spec)
            result = {
                "track": track,
                **asdict(spec),
                **metrics(valid["sold_price"].to_numpy(float), predicted),
                "seconds": round(time.perf_counter() - started, 2),
                "tree_count": model.tree_count_,
                "training_rows": len(spec_train),
            }
            rows.append(result)
            print(json.dumps(result))
    frame = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    frame.to_csv(RESULTS / "screen_november_2025.csv", index=False)
    print(frame[["track", "name", "mape", "mdape", "median_signed_error", "seconds"]].to_string(index=False))


def best_screen_specs() -> dict[str, ModelSpec]:
    path = RESULTS / "screen_november_2025.csv"
    if not path.exists():
        raise SystemExit("Run `screen` first.")
    results = pd.read_csv(path).sort_values(["track", "mdape", "mape"])
    selected = {}
    by_name = {spec.name: spec for spec in SPECS}
    for track in TRACKS:
        selected[track] = by_name[results.loc[results.track == track, "name"].iloc[0]]
    return selected


def walk_forward() -> None:
    """Run the handout's exact six-month walk-forward validation."""
    listings = data.load("listings")
    selected = best_screen_specs()
    prediction_frames = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for month in OFFICIAL_MONTHS:
        train = listings[listings["date_sold"] < month.start_time].copy()
        homes = listings[listings["date_sold"].dt.to_period("M") == month].copy()
        for track in TRACKS:
            spec = selected[track]
            spec_train = restrict_history(train, month.start_time, spec)
            model = fit_model(spec_train, track, spec)
            visible = data.hide_answers(homes, track)
            predicted = predict_model(model, visible, track, spec)
            prediction_frames.append(pd.DataFrame({
                "ml_num": homes["ml_num"].to_numpy(),
                "track": track,
                "month": str(month),
                "house_type_name": homes["house_type_name"].to_numpy(),
                "sold_price": homes["sold_price"].to_numpy(float),
                "predicted_price": predicted,
                "model": spec.name,
            }))
            score = metrics(homes["sold_price"].to_numpy(float), predicted)
            print(month, track, spec.name, json.dumps(score))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(RESULTS / "walk_forward_predictions.csv", index=False)
    report = {
        track: metrics(
            predictions.loc[predictions.track == track, "sold_price"].to_numpy(float),
            predictions.loc[predictions.track == track, "predicted_price"].to_numpy(float),
        )
        for track in TRACKS
    }
    (RESULTS / "walk_forward_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def train_final() -> None:
    """Fit selected models on all history and save models plus test predictions."""
    listings = data.load("listings")
    selected = best_screen_specs()
    timestamp = pd.Timestamp.now(tz="America/Toronto").strftime("%Y%m%d-%H%M%S")
    run_dir = ARTIFACTS / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": timestamp, "models": {}}
    for track in TRACKS:
        spec = selected[track]
        model = fit_model(restrict_history(listings, pd.Timestamp("2026-06-01"), spec), track, spec)
        model_path = run_dir / f"{track}.cbm"
        model.save_model(model_path)
        homes = data.load(track)
        predicted = predict_model(model, homes, track, spec)
        pd.DataFrame({"ml_num": homes["ml_num"], "predicted_price": np.round(predicted)}).to_csv(
            run_dir / f"submission_{track}.csv", index=False
        )
        manifest["models"][track] = {"spec": asdict(spec), "checkpoint": model_path.name}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved final isolated artifacts to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "walk-forward", "train-final"))
    return parser.parse_args()


if __name__ == "__main__":
    command = parse_args().command
    {"screen": screen, "walk-forward": walk_forward, "train-final": train_final}[command]()
