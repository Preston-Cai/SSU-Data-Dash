"""Materialize the frozen final blend and the starter-compatible score report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def main() -> None:
    base = pd.read_csv(RESULTS / "ensemble_walk_forward_predictions.csv")
    xgb = pd.read_csv(RESULTS / "xgboost_walk_forward_predictions.csv")

    off = base[base.track == "off_market"].merge(
        xgb[["ml_num", "month", "xgboost_prediction"]], on=["ml_num", "month"], validate="one_to_one"
    )
    off["predicted_price"] = geometric_blend(
        [off.recent_prediction, off.text_prediction, off.xgboost_prediction],
        [0.15, 0.20, 0.65],
    ) * 0.9772745745864185

    listed = base[base.track == "list_aware"].copy()
    listed["predicted_price"] = geometric_blend(
        [listed.regular_prediction, listed.recent_prediction, listed.text_prediction],
        [0.35, 0.35, 0.30],
    ) * 0.9865499704372777

    final = pd.concat([off[base.columns], listed[base.columns]], ignore_index=True)
    final.to_csv(RESULTS / "final_walk_forward_predictions.csv", index=False)

    report = {
        "validation_protocol": {
            "weight_selection_months": ["2025-09", "2025-10", "2025-11"],
            "untouched_official_months": ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"],
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
