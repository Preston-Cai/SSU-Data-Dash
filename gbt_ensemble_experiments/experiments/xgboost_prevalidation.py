"""September-October predictions for the selected off-market XGBoost."""

from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

BASE = runpy.run_path(str(HERE / "xgboost_walk_forward.py"), run_name="xgb_pre_base")
fit_predict = BASE["fit_predict"]
RESULTS = HERE.parent / "results"


def main():
    listings = data.load("listings")
    frames = []
    for month in pd.period_range("2025-09", "2025-10", freq="M"):
        homes = listings[listings.date_sold.dt.to_period("M") == month].copy()
        started = time.perf_counter()
        prediction = fit_predict(listings, homes, month.start_time)
        frames.append(pd.DataFrame({
            "ml_num": homes.ml_num.to_numpy(),
            "month": str(month),
            "sold_price": homes.sold_price.to_numpy(float),
            "xgboost_prediction": prediction,
        }))
        print(month, "seconds", round(time.perf_counter() - started, 2), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(
        RESULTS / "xgboost_prevalidation_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
