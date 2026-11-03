"""Inspect saved v8 and current validation predictions without training models."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import data
from features import square_feet


def main():
    root = Path(__file__).parent
    listings = data.load("listings")
    report = ["# Saved validation error analysis", "",
              "V8 is the reference. Current means the saved output file, not the current source code.",
              "Text associations are exploratory and may reflect property mix; they are not causal effects.", ""]
    for name, path in [("v8", root / "checkpoints/municipality_catboost_v8/validation_predictions.csv"),
                       ("current saved run", root / "outputs/validation_predictions.csv")]:
        predictions = pd.read_csv(path)
        extra = [c for c in listings.columns if c not in predictions.columns]
        rows = predictions.merge(listings[["ml_num"] + extra], on="ml_num", validate="many_to_one")
        rows["signed"] = 100 * (rows.predicted_price / rows.sold_price - 1)
        rows["ape"] = rows.signed.abs()
        rows["size_band"] = pd.cut(square_feet(rows.house_area),
                                   [0, 700, 1200, 1800, 2500, np.inf])
        rows["price_band"] = pd.qcut(rows.sold_price, 5, duplicates="drop")
        report += [f"## {name}", ""]
        for field in ["track", "house_type_name", "month", "size_band", "price_band", "municipality", "community"]:
            keys = ["track"] if field == "track" else ["track", field]
            table = rows.groupby(keys, observed=True).agg(
                n=("ape", "size"), MdAPE=("ape", "median"), MAPE=("ape", "mean"),
                median_signed=("signed", "median"), p90_APE=("ape", lambda x: x.quantile(.9)))
            if field == "community":
                table = table[table.n >= 100]
            report += [f"### {field}", "", "```", table.round(2).to_string(), "```", ""]
        if name != "v8":
            continue
        off = rows[rows.track == "off_market"]
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=100,
                                     max_df=.8, stop_words="english", binary=True)
        matrix = vectorizer.fit_transform(off.remarks.fillna(""))
        residual = np.log(off.sold_price / off.predicted_price).to_numpy()
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        effects = np.asarray(matrix.T @ residual).ravel() / counts * 100
        selected = np.argsort(abs(effects))[-30:][::-1]
        table = pd.DataFrame({"phrase": vectorizer.get_feature_names_out()[selected],
                              "n": counts[selected], "mean_log_actual_over_prediction_pct": effects[selected]})
        for label, mask in [("Dec-Feb", off.month <= "2026-02"), ("Mar-May", off.month > "2026-02")]:
            m = matrix[mask.to_numpy()][:, selected]
            n = np.asarray(m.sum(axis=0)).ravel()
            table[label] = np.asarray(m.T @ residual[mask.to_numpy()]).ravel() / np.maximum(n, 1) * 100
        report += ["### Text associations, off-market", "", "```", table.round(2).to_string(index=False), "```", ""]
    destination = root / "reports/v8_error_analysis.md"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text("\n".join(report), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
