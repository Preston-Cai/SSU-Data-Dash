"""Choose track-specific multiplicative calibration factors from validation predictions."""
from pathlib import Path

import numpy as np
import pandas as pd


FACTORS = np.arange(0.990, 1.011, 0.001)
PREDICTIONS = Path("outputs/validation_predictions.csv")
REPORT = Path("reports/calibration_tuning.md")


def scores(rows, factor):
    error = np.abs(factor * rows["predicted_price"] - rows["sold_price"]) / rows["sold_price"] * 100
    return float(np.median(error)), float(np.mean(error))


def main():
    predictions = pd.read_csv(PREDICTIONS)
    report = ["# Calibration tuning", "", "Factors were selected from walk-forward validation predictions.", ""]
    best_factors = {}
    for track in ["off_market", "list_aware"]:
        rows = predictions[predictions["track"] == track]
        results = [(factor, *scores(rows, factor)) for factor in FACTORS]
        best = min(results, key=lambda result: (result[1], result[2]))
        best_factors[track] = best[0]
        report += [
            f"## {track}", "",
            "| Factor | MdAPE | MAPE |",
            "|---:|---:|---:|",
            *[f"| {factor:.3f} | {mdape:.2f}% | {mape:.2f}% |" for factor, mdape, mape in results],
            "",
            f"Selected factor: **{best[0]:.3f}** (MdAPE {best[1]:.2f}%, MAPE {best[2]:.2f}%).",
            "",
        ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(report))
    print("\n".join(report))
    print("Best factors:", best_factors)


if __name__ == "__main__":
    main()
