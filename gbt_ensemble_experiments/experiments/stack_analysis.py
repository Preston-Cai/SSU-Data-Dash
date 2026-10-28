"""November-only three-source ensemble search."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def score(y, p):
    signed = (p - y) / y * 100
    return np.mean(np.abs(signed)), np.median(np.abs(signed)), np.median(signed)


def load_predictions():
    paths = {
        "tree": RESULTS / "lightgbm_screen_predictions.csv",
        "text": RESULTS / "linear_text_screen_predictions.csv",
        "encoded": RESULTS / "target_encoding_screen_predictions.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    extended = pd.read_csv(RESULTS / "text_extended_screen_predictions.csv")
    frames["text"] = frames["text"].merge(
        extended.drop(columns="sold_price"), on="ml_num", validate="one_to_one"
    )
    combined = frames["tree"]
    for name in ("text", "encoded"):
        combined = combined.merge(
            frames[name].drop(columns="sold_price"), on="ml_num", validate="one_to_one"
        )
    return frames, combined


def main():
    frames, frame = load_predictions()
    y = frame.sold_price.to_numpy(float)
    rows = []
    weight_pairs = [
        (tree_weight, text_weight, 1 - tree_weight - text_weight)
        for tree_weight in np.linspace(0, 1, 11)
        for text_weight in np.linspace(0, 1, 11)
        if tree_weight + text_weight <= 1.00001
    ]
    for track in ("off_market", "list_aware"):
        columns = {
            source: [column for column in source_frame if column.startswith(track + "_")]
            for source, source_frame in frames.items()
        }
        for tree, text, encoded in itertools.product(
            columns["tree"], columns["text"], columns["encoded"]
        ):
            logs = [np.log(frame[column].to_numpy(float)) for column in (tree, text, encoded)]
            for tree_weight, text_weight, encoded_weight in weight_pairs:
                prediction = np.exp(
                    tree_weight * logs[0]
                    + text_weight * logs[1]
                    + encoded_weight * logs[2]
                )
                calibration = float(np.exp(np.median(np.log(y / prediction))))
                prediction *= calibration
                mape, mdape, signed = score(y, prediction)
                rows.append({
                    "track": track,
                    "tree_model": tree,
                    "text_model": text,
                    "encoded_model": encoded,
                    "tree_weight": tree_weight,
                    "text_weight": text_weight,
                    "encoded_weight": encoded_weight,
                    "calibration": calibration,
                    "mape": mape,
                    "mdape": mdape,
                    "median_signed_error": signed,
                })
    results = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    results.to_csv(RESULTS / "stack_screen_november_2025.csv", index=False)
    print(results.groupby("track", sort=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
