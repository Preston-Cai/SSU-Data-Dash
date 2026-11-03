"""Grouped permutation importance on the final walk-forward validation month."""
import numpy as np
import pandas as pd

import data
import models


VALIDATION_MONTH = pd.Period("2026-05", freq="M")
REPEATS = 5


def mdape(predicted, actual):
    return float(np.median(np.abs(predicted - actual) / actual) * 100)


def feature_group(name):
    if name.startswith("community_"):
        return "community"
    if name.startswith("house_style_"):
        return "house_style"
    if name in {
        "prior_log_price_per_sqft",
        "prior_sales_count",
        "recent_log_price_per_sqft",
        "recent_sales_count",
    }:
        return "historical_comparables"
    if name in {
        "spatial_recent_log_price_per_sqft",
        "spatial_recent_sales_count",
        "nearest_comparable_distance",
    }:
        return "spatial_comparables"
    if name.startswith("is_"):
        return "property_type"
    return name


def evaluate_track(listings, track):
    train = listings[listings["date_sold"] < VALIDATION_MONTH.start_time]
    homes = listings[listings["date_sold"].dt.to_period("M") == VALIDATION_MONTH]
    homes = data.hide_answers(homes, track)

    model = models.make_model(track).fit(train)
    raw = model.make_features(homes)
    enriched = model._add_history_features(
        raw, homes["date_listed"], homes["community"], homes["house_type_name"],
        homes["latitude"], homes["longitude"]
    )
    transformed = model._transform(enriched)
    actual = listings.loc[homes.index, "sold_price"].to_numpy(dtype=float)
    baseline = mdape(np.exp(model.model.predict(transformed)), actual)

    names = list(model.numeric_columns)
    if model.categorical_columns:
        names += list(model.encoder.get_feature_names_out(model.categorical_columns))

    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(feature_group(name), []).append(index)

    rng = np.random.default_rng(0)
    rows = []
    for group, indexes in groups.items():
        scores = []
        for _ in range(REPEATS):
            permutation = rng.permutation(len(transformed))
            shuffled = transformed.copy()
            shuffled[:, indexes] = transformed[permutation][:, indexes]
            score = mdape(np.exp(model.model.predict(shuffled)), actual)
            scores.append(score)
        rows.append({
            "feature": group,
            "baseline_mdape": baseline,
            "mdape_after_permutation": float(np.mean(scores)),
            "importance_pp": float(np.mean(scores) - baseline),
            "std_pp": float(np.std(scores)),
        })
    return baseline, sorted(rows, key=lambda row: row["importance_pp"], reverse=True)


def main():
    listings = data.load("listings")
    report = [
        "# Grouped permutation importance",
        "",
        f"Future holdout month: {VALIDATION_MONTH}",
        f"Each feature group was shuffled {REPEATS} times after training on earlier sales.",
        "Importance is the increase in MdAPE, in percentage points; larger positive values mean the model relied more on that group.",
        "",
    ]
    for track in ["off_market", "list_aware"]:
        baseline, rows = evaluate_track(listings, track)
        report += [f"## {track}", "", f"Baseline MdAPE: **{baseline:.2f}%**", "",
                   "| Feature group | MdAPE after shuffle | Importance (pp) | Std. dev. (pp) |",
                   "|---|---:|---:|---:|"]
        for row in rows:
            report.append(f"| {row['feature']} | {row['mdape_after_permutation']:.2f}% | {row['importance_pp']:+.2f} | {row['std_pp']:.2f} |")
        report.append("")

    output = "\n".join(report)
    print(output)
    (data.Path("reports") / "permutation_importance.md").write_text(output + "\n")


if __name__ == "__main__":
    main()
