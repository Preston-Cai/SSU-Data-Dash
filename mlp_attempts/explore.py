"""Exploratory plots for the historical home-sales data."""

from pathlib import Path
import sys

import matplotlib

# Save figures reliably when this script is run from a terminal or CI session.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import data


RANDOM_SEED = 42
MAX_SCATTER_ROWS = 1_000
OUTPUT_DIR = ROOT / "figures" / "exploration"


def sample_rows(frame: pd.DataFrame, limit: int = MAX_SCATTER_ROWS) -> pd.DataFrame:
    """Return at most ``limit`` rows, sampled reproducibly."""
    if len(frame) <= limit:
        return frame.copy()
    return frame.sample(n=limit, random_state=RANDOM_SEED)


def house_area_midpoint(value: object) -> float:
    """Convert an area value or range such as '700-899' to square feet."""
    numbers = pd.to_numeric(
        pd.Series(str(value).replace(",", "").split("-")), errors="coerce"
    ).dropna()
    return float(numbers.mean()) if not numbers.empty else np.nan


def dollars(value: float, _position: float) -> str:
    return f"${value / 1_000_000:.1f}M"


def save_municipality_price_ranges(listings: pd.DataFrame) -> Path:
    summary = (
        listings.groupby("municipality", observed=True)["sold_price"]
        .agg(minimum="min", median="median", maximum="max")
        .sort_values("median", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(summary))
    ax.hlines(
        positions,
        summary["minimum"],
        summary["maximum"],
        color="#4472C4",
        linewidth=5,
        alpha=0.75,
        label="Minimum to maximum",
    )
    ax.scatter(
        summary["minimum"], positions, color="#264653", marker="|", s=180, zorder=3
    )
    ax.scatter(
        summary["maximum"], positions, color="#264653", marker="|", s=180, zorder=3
    )
    ax.scatter(
        summary["median"],
        positions,
        color="#E76F51",
        s=70,
        zorder=4,
        label="Median",
    )
    ax.set_title("Sold-price range by municipality")
    ax.set_xlabel("Sold price (log scale)")
    ax.set_ylabel("Municipality")
    ax.set_yticks(positions, labels=summary.index)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(dollars))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = OUTPUT_DIR / "price-range-by-municipality.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_area_vs_price(listings: pd.DataFrame) -> Path:
    plot_data = listings[["house_area", "sold_price"]].copy()
    plot_data["house_area_sqft"] = plot_data["house_area"].map(house_area_midpoint)
    plot_data = sample_rows(plot_data.dropna(subset=["house_area_sqft", "sold_price"]))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        plot_data["house_area_sqft"],
        plot_data["sold_price"],
        s=24,
        alpha=0.45,
        color="#2A9D8F",
        edgecolors="none",
    )
    ax.set_title(f"House area vs. sold price (random sample, n={len(plot_data):,})")
    ax.set_xlabel("House area (sq. ft.; midpoint when reported as a range)")
    ax.set_ylabel("Sold price")
    ax.yaxis.set_major_formatter(FuncFormatter(dollars))
    ax.grid(alpha=0.2)
    fig.tight_layout()

    path = OUTPUT_DIR / "house-area-vs-sold-price.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_days_vs_price_difference(listings: pd.DataFrame) -> Path:
    plot_data = listings[["days_on_market", "sold_price", "list_price"]].copy()
    plot_data["price_difference"] = plot_data["sold_price"] - plot_data["list_price"]
    plot_data = sample_rows(plot_data.dropna())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        plot_data["days_on_market"],
        plot_data["price_difference"],
        s=24,
        alpha=0.45,
        color="#E76F51",
        edgecolors="none",
    )
    ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_title(
        f"Days on market vs. sold-minus-list price (random sample, n={len(plot_data):,})"
    )
    ax.set_xlabel("Days on market")
    ax.set_ylabel("Sold price - list price")
    ax.yaxis.set_major_formatter(FuncFormatter(dollars))
    ax.grid(alpha=0.2)
    fig.tight_layout()

    path = OUTPUT_DIR / "days-on-market-vs-price-difference.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_avm_estimate_timing(listings: pd.DataFrame) -> Path:
    estimate_date = listings["external_avm_estimate_date"]
    list_date = listings["date_listed"]
    sold_date = listings["date_sold"]
    labels = ["Before list date", "Between list and sold", "After sold date"]

    timing = np.select(
        [
            estimate_date < list_date,
            estimate_date.between(list_date, sold_date, inclusive="both"),
            estimate_date > sold_date,
        ],
        labels,
        default="Missing/invalid date",
    )
    counts = pd.Series(timing).value_counts().reindex(labels, fill_value=0)
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(counts.index, counts.values, color="#6C5CE7", width=0.7)
    ax.bar_label(
        bars,
        labels=[f"{count:,}\n({percent:.1f}%)" for count, percent in zip(counts, percentages)],
        padding=4,
    )
    ax.set_title("Timing of external AVM estimate")
    ax.set_xlabel("Estimate date relative to listing and sale")
    ax.set_ylabel("Number of sales")
    ax.set_ylim(0, counts.max() * 1.13)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = OUTPUT_DIR / "external-avm-estimate-timing.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    listings = data.load("listings")
    print(
        f"Loaded {len(listings):,} sales, "
        f"{listings['date_sold'].min():%B %Y} to {listings['date_sold'].max():%B %Y}."
    )
    
    off_market = data.load("off_market")
    list_aware = data.load("list_aware")
    print(f"Columns for off market: {off_market.columns.to_list()}")
    print(f"Columns for list aware: {list_aware.columns.to_list()}")


    # paths = [
    #     save_municipality_price_ranges(listings),
    #     save_area_vs_price(listings),
    #     save_days_vs_price_difference(listings),
    #     save_avm_estimate_timing(listings),
    # ]
    # for path in paths:
    #     print(f"Saved {path.relative_to(Path(__file__).resolve().parent)}")


if __name__ == "__main__":
    main()
