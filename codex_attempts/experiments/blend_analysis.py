"""Select November-only ensemble weights for isolated candidate models."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def score(actual, predicted):
    error = np.abs(predicted - actual) / actual * 100
    signed = (predicted - actual) / actual * 100
    return float(error.mean()), float(np.median(error)), float(np.median(signed))


def main():
    trees = pd.read_csv(RESULTS / "lightgbm_screen_predictions.csv")
    text = pd.read_csv(RESULTS / "linear_text_screen_predictions.csv")
    frame = trees.merge(text, on=["ml_num", "sold_price"], validate="one_to_one")
    rows = []
    for track in ("off_market", "list_aware"):
        tree_columns = [column for column in trees if column.startswith(track + "_")]
        text_columns = [column for column in text if column.startswith(track + "_")]
        actual = frame.sold_price.to_numpy(float)
        for tree_column, text_column, tree_weight in itertools.product(
            tree_columns, text_columns, np.linspace(0, 1, 21)
        ):
            # Geometric blending is natural because both models learn logarithmic
            # price targets and percentage error is the competition metric.
            predicted = np.exp(
                tree_weight * np.log(frame[tree_column].to_numpy(float))
                + (1 - tree_weight) * np.log(frame[text_column].to_numpy(float))
            )
            calibration = float(np.exp(np.median(np.log(actual / predicted))))
            for calibrated in (False, True):
                candidate = predicted * (calibration if calibrated else 1.0)
                mape, mdape, signed = score(actual, candidate)
                rows.append({
                    "track": track,
                    "tree_model": tree_column,
                    "text_model": text_column,
                    "tree_weight": tree_weight,
                    "calibrated": calibrated,
                    "calibration": calibration if calibrated else 1.0,
                    "mape": mape,
                    "mdape": mdape,
                    "median_signed_error": signed,
                })
    results = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    results.to_csv(RESULTS / "blend_screen_november_2025.csv", index=False)
    print(results.groupby("track", sort=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
