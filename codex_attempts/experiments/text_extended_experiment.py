"""Longer-running text candidates retained for ensemble testing."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

BASE = runpy.run_path(str(HERE / "linear_text_experiment.py"), run_name="text_extended_base")
COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="text_extended_common")
SparseFeatures = BASE["SparseFeatures"]
target = BASE["target"]
invert = BASE["invert"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


VARIANTS = (
    {"name": "word_all_40k", "analyzer": "word", "months": None, "max_features": 40_000, "min_df": 5},
    {"name": "char_h36_30k", "analyzer": "char", "months": 36, "max_features": 30_000, "min_df": 10},
)


def main():
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    output = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    rows = []
    for track in ("off_market", "list_aware"):
        visible = data.hide_answers(valid, track)
        for variant in VARIANTS:
            train = listings[listings.date_sold < cutoff].copy()
            if variant["months"] is not None:
                train = train[train.date_sold >= cutoff - pd.DateOffset(months=variant["months"])].copy()
            started = time.perf_counter()
            builder = SparseFeatures(
                track,
                variant["analyzer"],
                max_features=variant["max_features"],
                min_df=variant["min_df"],
            )
            x_train = builder.fit_transform(train)
            x_valid = builder.transform(visible)
            print(track, variant["name"], x_train.shape, x_train.nnz, flush=True)
            for alpha in (1.0, 10.0, 50.0, 200.0):
                model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=2500)
                model.fit(x_train, target(train, track))
                predicted = invert(model.predict(x_valid), visible, track)
                result = {
                    "track": track,
                    "variant": variant["name"],
                    "alpha": alpha,
                    **metrics(valid.sold_price.to_numpy(float), predicted),
                    "training_rows": len(train),
                    "features": x_train.shape[1],
                    "seconds_cumulative": round(time.perf_counter() - started, 2),
                }
                rows.append(result)
                output[f"{track}_{variant['name']}_a{alpha:g}"] = predicted
                print(json.dumps(result), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    frame.to_csv(RESULTS / "text_extended_screen_november_2025.csv", index=False)
    output.to_csv(RESULTS / "text_extended_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
