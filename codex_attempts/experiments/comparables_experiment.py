"""Tune a leakage-safe comparable-sales estimator on November 2025."""

from __future__ import annotations

import itertools
import json
import runpy
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="comparable_common")
square_feet = COMMON["square_feet"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


@dataclass(frozen=True)
class CompSpec:
    group: str
    history_months: int
    k: int
    geo_scale: float
    area_scale: float
    time_scale: float
    price_mode: str
    weighted: bool = True

    @property
    def name(self) -> str:
        return (
            f"comp_{self.group}_h{self.history_months}_k{self.k}_g{self.geo_scale}"
            f"_a{self.area_scale}_t{self.time_scale}_{self.price_mode}"
        )


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=rows.index)
    frame["house_type_name"] = rows.house_type_name.astype(str)
    frame["community"] = rows.community.astype(str)
    frame["house_style"] = rows.house_style.astype(str)
    frame["latitude"] = rows.latitude.astype(float)
    frame["longitude"] = rows.longitude.astype(float)
    frame["area"] = square_feet(rows.house_area).clip(lower=100).astype(float)
    frame["log_area"] = np.log(frame.area)
    frame["bedrooms"] = rows.bedrooms.astype(float)
    frame["washrooms"] = rows.washrooms.astype(float)
    frame["parking"] = rows.parking_total.astype(float)
    frame["garage"] = rows.garage.astype(float)
    frame["month"] = (pd.to_datetime(rows.date_listed).dt.year - 2021) * 12 + pd.to_datetime(rows.date_listed).dt.month - 1
    if "date_sold" in rows:
        frame["date_sold"] = pd.to_datetime(rows.date_sold)
        frame["sale_month"] = (frame.date_sold.dt.year - 2021) * 12 + frame.date_sold.dt.month - 1
    if "sold_price" in rows:
        frame["sold_price"] = rows.sold_price.astype(float)
        frame["log_price"] = np.log(frame.sold_price)
        frame["log_ppsf"] = np.log(frame.sold_price / frame.area)
    return frame


class ComparableModel:
    def __init__(self, spec: CompSpec):
        self.spec = spec

    def fit(self, sales: pd.DataFrame, cutoff: pd.Timestamp):
        earliest = cutoff - pd.DateOffset(months=self.spec.history_months)
        eligible = sales[(sales.date_sold < cutoff) & (sales.date_sold >= earliest)].copy()
        self.sales = prepare(eligible).reset_index(drop=True)
        if self.spec.group == "community_type":
            keys = ["community", "house_type_name"]
        elif self.spec.group == "community":
            keys = ["community"]
        else:
            keys = ["house_type_name"]
        self.keys = keys
        self.groups = {key if isinstance(key, tuple) else (key,): idx.to_numpy() for key, idx in self.sales.groupby(keys, observed=True).groups.items()}
        return self

    def predict(self, homes: pd.DataFrame) -> np.ndarray:
        targets = prepare(homes).reset_index(drop=True)
        global_by_type = {
            key: idx.to_numpy()
            for key, idx in self.sales.groupby("house_type_name", observed=True).groups.items()
        }
        result = np.empty(len(targets), dtype=float)
        for position, home in targets.iterrows():
            key = tuple(home[column] for column in self.keys)
            candidates = self.groups.get(key)
            if candidates is None or len(candidates) < max(5, self.spec.k // 2):
                candidates = global_by_type.get(home.house_type_name, self.sales.index.to_numpy())
            comp = self.sales.iloc[candidates]
            dx = (comp.longitude.to_numpy() - home.longitude) * 0.72 / self.spec.geo_scale
            dy = (comp.latitude.to_numpy() - home.latitude) / self.spec.geo_scale
            da = (comp.log_area.to_numpy() - home.log_area) / self.spec.area_scale
            db = (comp.bedrooms.to_numpy() - home.bedrooms) / 1.5
            dw = (comp.washrooms.to_numpy() - home.washrooms) / 1.5
            dp = (comp.parking.to_numpy() - home.parking) / 3.0
            dg = (comp.garage.to_numpy() - home.garage) / 2.0
            dt = (home.month - comp.sale_month.to_numpy()) / self.spec.time_scale
            style = (comp.house_style.to_numpy() != home.house_style).astype(float) * 0.5
            distance2 = dx * dx + dy * dy + da * da + db * db + dw * dw + dp * dp + dg * dg + dt * dt + style
            count = min(self.spec.k, len(comp))
            nearest = np.argpartition(distance2, count - 1)[:count]
            selected_d2 = distance2[nearest]
            if self.spec.price_mode == "ppsf":
                values = comp.log_ppsf.to_numpy()[nearest] + home.log_area
            elif self.spec.price_mode == "hybrid":
                direct = comp.log_price.to_numpy()[nearest]
                ppsf = comp.log_ppsf.to_numpy()[nearest] + home.log_area
                values = 0.5 * direct + 0.5 * ppsf
            else:
                values = comp.log_price.to_numpy()[nearest]
            if self.spec.weighted:
                weights = 1.0 / np.maximum(np.sqrt(selected_d2), 0.15)
                raw = np.sum(weights * values) / np.sum(weights)
            else:
                raw = np.median(values)
            result[position] = np.exp(raw)
        return result


def specs():
    # A compact, structured grid; every choice is fixed before seeing November.
    for group, history, k, geo, area, time_scale, price_mode in itertools.product(
        ("community_type", "type"),
        (24, 36),
        (10, 25),
        (0.01, 0.03),
        (0.35,),
        (12.0,),
        ("hybrid",),
    ):
        yield CompSpec(group, history, k, geo, area, time_scale, price_mode)


def screen() -> None:
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    rows = []
    started = time.perf_counter()
    for number, spec in enumerate(specs(), 1):
        model = ComparableModel(spec).fit(listings, cutoff)
        predicted = model.predict(data.hide_answers(valid, "off_market"))
        score = metrics(valid.sold_price.to_numpy(float), predicted)
        rows.append({**asdict(spec), "name": spec.name, **score})
        if number % 4 == 0:
            best = min(rows, key=lambda row: (row["mdape"], row["mape"]))
            print(number, json.dumps({k: best[k] for k in ("name", "mape", "mdape", "median_signed_error")}), flush=True)
    frame = pd.DataFrame(rows).sort_values(["mdape", "mape"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "comparables_screen_november_2025.csv", index=False)
    print(frame.head(20).to_string(index=False))
    print(f"Grid seconds: {time.perf_counter() - started:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen",))
    return parser.parse_args()


if __name__ == "__main__":
    screen()
