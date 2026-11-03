"""Rank numeric variables by correlation with list_price and plot the strongest ones."""
from pathlib import Path
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data


ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--Baliyang--SSU_data_dash" / "snapshots"


def parse_area(series):
    """Convert ranges such as '700-899' to their midpoint."""
    parts = series.astype("string").str.replace(",", "", regex=False).str.extract(
        r"(\d+(?:\.\d+)?)[^\d]+(\d+(?:\.\d+)?)"
    )
    return parts.astype(float).mean(axis=1)


def correlations(df):
    target = df["list_price"]
    candidates = df.select_dtypes(include="number").drop(columns=["list_price"], errors="ignore").copy()
    candidates["house_area_mid"] = parse_area(df["house_area"])
    rows = []
    for name in candidates:
        x = candidates[name]
        valid = x.notna() & target.notna()
        if valid.sum() < 10:
            continue
        rows.append({
            "variable": name,
            "n": int(valid.sum()),
            "pearson": x[valid].corr(target[valid], method="pearson"),
            "spearman": x[valid].corr(target[valid], method="spearman"),
        })
    return pd.DataFrame(rows).sort_values("pearson", key=lambda s: s.abs(), ascending=False)


def scatterplots(df, variables, path):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    target = df["list_price"]
    for ax, name in zip(axes.ravel(), variables):
        x = parse_area(df["house_area"]) if name == "house_area_mid" else df[name]
        valid = x.notna() & target.notna() & (target > 0)
        # A random sample keeps the plots readable while preserving the overall relationship.
        idx = np.flatnonzero(valid.to_numpy())
        if len(idx) > 15000:
            idx = rng.choice(idx, 15000, replace=False)
        xv, yv = x.iloc[idx], target.iloc[idx]
        ax.scatter(xv, yv, s=6, alpha=0.15, color="#1769aa", edgecolors="none", rasterized=True)
        if xv.nunique() > 1:
            z = np.polyfit(xv, yv, 1)
            xx = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xx, z[0] * xx + z[1], color="#d1495b", linewidth=2)
        corr = xv.corr(yv)
        ax.set_title(f"{name.replace('_', ' ').title()}  (r = {corr:.3f})")
        ax.set_xlabel(name.replace("_", " ").title())
        ax.set_ylabel("List price ($)")
        ax.yaxis.set_major_formatter(lambda value, _: f"${value / 1e6:.1f}M")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(variables):]:
        ax.set_visible(False)
    fig.suptitle("Variables most correlated with listed price\n(training listings; red line = linear trend)", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    # data.py uses DATA_DIR when supplied; use the existing local cache when available.
    if "DATA_DIR" not in os.environ and DEFAULT_DATA_DIR.exists():
        snapshots = sorted(DEFAULT_DATA_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            os.environ["DATA_DIR"] = str(snapshots[0])
    df = data.load("listings")
    result = correlations(df)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    result.to_csv(out / "list_price_correlations.csv", index=False)

    usable = [
        "house_area_mid", "washrooms", "bedrooms", "garage", "parking_total", "lot_depth"
    ]
    scatterplots(df, usable, out / "list_price_scatterplots.png")
    print(result.to_string(index=False))
    print(f"\nWrote {out / 'list_price_correlations.csv'}")
    print(f"Wrote {out / 'list_price_scatterplots.png'}")


if __name__ == "__main__":
    main()
