"""Reproduce final ensemble weights and calibration from Sep-Nov 2025.

The search uses only prevalidation months, never the official Dec-May window.
Weights lie on a 0.05 grid and sum to one. Candidates are ranked by pooled
MdAPE, then mean monthly MdAPE, then MAPE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def geometric_blend(frame: pd.DataFrame, columns: list[str], weights: list[float]) -> np.ndarray:
    """Weighted average in log-price space, returned in dollars."""
    return np.exp(
        sum(weight * np.log(frame[column].to_numpy(float)) for column, weight in zip(columns, weights))
    )


def multiplicative_calibration(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Center the median log(actual/predicted) residual at zero."""
    return float(np.exp(np.median(np.log(actual / predicted))))


def build_candidate_frames() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Join the exact finalist component predictions for Sep-Nov 2025."""
    pre = pd.read_csv(RESULTS / "ensemble_prevalidation_predictions.csv")
    xgb_pre = pd.read_csv(RESULTS / "xgboost_prevalidation_predictions.csv")
    recent = pd.read_csv(RESULTS / "recency_encoding_screen_predictions.csv")
    text = pd.read_csv(RESULTS / "linear_text_screen_predictions.csv")
    xgb_nov = pd.read_csv(RESULTS / "xgboost_screen_predictions.csv")
    encoded = pd.read_csv(RESULTS / "target_encoding_screen_predictions.csv")

    off_early = pre[pre.track == "off_market"].merge(
        xgb_pre, on=["ml_num", "month", "sold_price"], validate="one_to_one"
    )
    off_nov = (
        recent[["ml_num", "sold_price", "off_market_recent_hl24"]]
        .merge(
            text[["ml_num", "sold_price", "off_market_word_ridge_a1"]],
            on=["ml_num", "sold_price"],
            validate="one_to_one",
        )
        .merge(
            xgb_nov[["ml_num", "sold_price", "off_market_xgb_d8_squarederror"]],
            on=["ml_num", "sold_price"],
            validate="one_to_one",
        )
        .rename(
            columns={
                "off_market_recent_hl24": "recent_prediction",
                "off_market_word_ridge_a1": "text_prediction",
                "off_market_xgb_d8_squarederror": "xgboost_prediction",
            }
        )
    )
    off_nov["month"] = "2025-11"
    off_columns = ["recent_prediction", "text_prediction", "xgboost_prediction"]
    off = pd.concat(
        [
            off_early[["ml_num", "month", "sold_price", *off_columns]],
            off_nov[["ml_num", "month", "sold_price", *off_columns]],
        ],
        ignore_index=True,
    )

    list_early = pre[pre.track == "list_aware"].copy()
    list_nov = (
        encoded[["ml_num", "sold_price", "list_aware_te_p5_regression_l1"]]
        .merge(
            recent[["ml_num", "sold_price", "list_aware_recent_hl4"]],
            on=["ml_num", "sold_price"],
            validate="one_to_one",
        )
        .merge(
            text[["ml_num", "sold_price", "list_aware_word_ridge_a50"]],
            on=["ml_num", "sold_price"],
            validate="one_to_one",
        )
        .rename(
            columns={
                "list_aware_te_p5_regression_l1": "regular_prediction",
                "list_aware_recent_hl4": "recent_prediction",
                "list_aware_word_ridge_a50": "text_prediction",
            }
        )
    )
    list_nov["month"] = "2025-11"
    list_columns = ["regular_prediction", "recent_prediction", "text_prediction"]
    listed = pd.concat(
        [
            list_early[["ml_num", "month", "sold_price", *list_columns]],
            list_nov[["ml_num", "month", "sold_price", *list_columns]],
        ],
        ignore_index=True,
    )
    return {
        "off_market": (off, off_columns),
        "list_aware": (listed, list_columns),
    }


def search_weights(
    frame: pd.DataFrame, component_columns: list[str], step: float = 0.05
) -> pd.DataFrame:
    """Evaluate every three-component weight combination on the simplex."""
    if len(component_columns) != 3:
        raise ValueError("This search expects exactly three components.")
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0 exactly.")

    actual = frame.sold_price.to_numpy(float)
    rows = []
    for first_units in range(units + 1):
        for second_units in range(units - first_units + 1):
            weights = [
                first_units / units,
                second_units / units,
                (units - first_units - second_units) / units,
            ]
            raw = geometric_blend(frame, component_columns, weights)
            calibration = multiplicative_calibration(actual, raw)
            predicted = raw * calibration
            absolute_error = np.abs(predicted - actual) / actual * 100
            monthly_mdape = [
                float(np.median(absolute_error[frame.month.to_numpy() == month]))
                for month in sorted(frame.month.unique())
            ]
            rows.append(
                {
                    **{f"weight_{name}": value for name, value in zip(component_columns, weights)},
                    "calibration": calibration,
                    "mdape": float(np.median(absolute_error)),
                    "mean_monthly_mdape": float(np.mean(monthly_mdape)),
                    "mape": float(np.mean(absolute_error)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["mdape", "mean_monthly_mdape", "mape"], ignore_index=True
    )


def main() -> None:
    all_candidates = []
    selected = {}
    for track, (frame, columns) in build_candidate_frames().items():
        candidates = search_weights(frame, columns)
        candidates.insert(0, "track", track)
        all_candidates.append(candidates)
        winner = candidates.iloc[0]
        selected[track] = {
            "components": columns,
            "weights": [float(winner[f"weight_{column}"]) for column in columns],
            "calibration": float(winner.calibration),
            "prevalidation_mdape": float(winner.mdape),
            "prevalidation_mape": float(winner.mape),
            "rows": len(frame),
            "months": sorted(frame.month.unique().tolist()),
        }

    pd.concat(all_candidates, ignore_index=True).to_csv(
        RESULTS / "final_blend_weight_search.csv", index=False
    )
    (RESULTS / "final_blend_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
