"""Run the frozen ensemble components on September-October 2025."""

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

BASE = runpy.run_path(str(HERE / "ensemble_walk_forward.py"), run_name="prevalidation_base")
component_predictions = BASE["component_predictions"]
RESULTS = HERE.parent / "results"


def main():
    listings = data.load("listings")
    frames = []
    for month in pd.period_range("2025-09", "2025-10", freq="M"):
        homes = listings[listings.date_sold.dt.to_period("M") == month].copy()
        for track in ("off_market", "list_aware"):
            started = time.perf_counter()
            regular, recent, text = component_predictions(listings, homes, track, month.start_time)
            frames.append(pd.DataFrame({
                "ml_num": homes.ml_num.to_numpy(),
                "track": track,
                "month": str(month),
                "sold_price": homes.sold_price.to_numpy(float),
                "regular_prediction": regular,
                "recent_prediction": recent,
                "text_prediction": text,
            }))
            print(month, track, "seconds", round(time.perf_counter() - started, 2), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(
        RESULTS / "ensemble_prevalidation_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
