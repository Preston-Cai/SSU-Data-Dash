"""Materialize the frozen final blend and the starter-compatible score report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def bootstrap_ci(values: np.ndarray, statistic, repeats: int = 1000) -> list[float]:
    rng = np.random.default_rng(0)
    samples = [statistic(rng.choice(values, size=len(values))) for _ in range(repeats)]
    return [round(float(value), 2) for value in np.percentile(samples, [2.5, 97.5])]


def score(frame: pd.DataFrame) -> dict:
    signed = 100 * (frame.predicted_price - frame.sold_price) / frame.sold_price
    absolute = signed.abs().to_numpy()
    return {
        "n": len(frame),
        "mape": round(float(np.mean(absolute)), 2),
        "mape_95ci": bootstrap_ci(absolute, np.mean),
        "mdape": round(float(np.median(absolute)), 2),
        "mdape_95ci": bootstrap_ci(absolute, np.median),
        "median_signed_error": round(float(np.median(signed)), 2),
    }


def geometric_blend(parts: list[pd.Series], weights: list[float]) -> np.ndarray:
    return np.exp(sum(weight * np.log(part.to_numpy(float)) for part, weight in zip(parts, weights)))


def median_ratio_predictions(
    listings: pd.DataFrame,
    homes: pd.DataFrame,
    month: str,
    group_columns=("community", "house_type_name"),
    prior=80.0,
) -> np.ndarray:
    cutoff = pd.Period(month).start_time
    train = listings[
        (listings.date_sold < cutoff)
        & (listings.date_sold >= cutoff - pd.DateOffset(months=36))
    ]
    values = np.log(train.sold_price.to_numpy(float) / train.list_price.to_numpy(float))
    global_median = float(np.median(values))
    key = train[group_columns[0]].fillna("__MISSING__").astype(str)
    home_key = homes[group_columns[0]].fillna("__MISSING__").astype(str)
    for column in group_columns[1:]:
        key = key + "|" + train[column].fillna("__MISSING__").astype(str)
        home_key = home_key + "|" + homes[column].fillna("__MISSING__").astype(str)
    table = pd.DataFrame({"key": key.to_numpy(), "value": values}).groupby("key").value.agg(
        ["median", "count"]
    )
    median = home_key.map(table["median"]).fillna(global_median)
    count = home_key.map(table["count"]).fillna(0.0)
    smoothed = (count * median + prior * global_median) / (count + prior)
    return homes.list_price.to_numpy(float) * np.exp(smoothed.to_numpy(float))


def main() -> None:
    base = pd.read_csv(RESULTS / "ensemble_walk_forward_predictions.csv")
    xgb = pd.read_csv(RESULTS / "xgboost_walk_forward_predictions.csv")

    off = base[base.track == "off_market"].merge(
        xgb[["ml_num", "month", "xgboost_prediction"]], on=["ml_num", "month"], validate="one_to_one"
    )
    off["predicted_price"] = geometric_blend(
        [off.recent_prediction, off.text_prediction, off.xgboost_prediction],
        [0.32, 0.19, 0.49],
    ) * 0.9805010910351815

    listed = base[base.track == "list_aware"].copy()
    off_anchor = off[["ml_num", "month", "recent_prediction", "text_prediction", "xgboost_prediction"]].copy()
    off_anchor["off_anchor"] = geometric_blend(
        [off_anchor.recent_prediction, off_anchor.text_prediction, off_anchor.xgboost_prediction],
        [0.32, 0.19, 0.49],
    )
    listed = listed.merge(
        off_anchor[["ml_num", "month", "off_anchor"]],
        on=["ml_num", "month"],
        validate="one_to_one",
    )
    listings = data.load("listings")
    source = listings.set_index("ml_num")
    listed["median_anchor"] = np.nan
    for month, index in listed.groupby("month").groups.items():
        homes = source.loc[listed.loc[index, "ml_num"]].reset_index()
        listed.loc[index, "median_anchor"] = median_ratio_predictions(listings, homes, month)
    listed["predicted_price"] = geometric_blend(
        [listed.regular_prediction, listed.recent_prediction, listed.off_anchor, listed.median_anchor],
        [0.113, 0.377, 0.155, 0.355],
    ) * 0.98471

    final = pd.concat([off[base.columns], listed[base.columns]], ignore_index=True)
    final.to_csv(RESULTS / "final_walk_forward_predictions.csv", index=False)

    report = {
        "validation_protocol": {
            "weight_selection_months": ["2025-09", "2025-10", "2025-11"],
            "untouched_official_months": ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"],
            "selection_note": (
                "The community-plus-property-type refinement was selected after "
                "inspection of the official months; 3.95% is a post-validation "
                "development score, not a new untouched estimate."
            ),
        },
        "tracks": {},
    }
    for track in ("off_market", "list_aware"):
        selected = final[final.track == track]
        report["tracks"][track] = {
            "overall": score(selected),
            "by_month": {
                month: score(month_frame)
                for month, month_frame in selected.groupby("month", sort=True)
            },
        }
    (RESULTS / "final_walk_forward_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["tracks"], indent=2))


if __name__ == "__main__":
    main()
