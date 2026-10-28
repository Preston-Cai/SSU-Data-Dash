"""Sparse hedonic + listing-text models tuned before official validation."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

COMMON = runpy.run_path(str(HERE / "experiment.py"), run_name="linear_common")
make_raw_features = COMMON["make_features"]
metrics = COMMON["metrics"]
RESULTS = HERE.parent / "results"


class SparseFeatures:
    def __init__(
        self,
        track: str,
        analyzer: str,
        *,
        max_features: int | None = None,
        min_df: int | None = None,
    ):
        self.track = track
        self.analyzer = analyzer
        if analyzer == "word":
            self.text = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=min_df or 10,
                max_df=0.995,
                max_features=max_features or 20_000,
                sublinear_tf=True,
                dtype=np.float32,
            )
        else:
            self.text = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=min_df or 10,
                max_features=max_features or 40_000,
                sublinear_tf=True,
                dtype=np.float32,
            )
        self.one_hot = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        self.scaler = StandardScaler()

    def _parts(self, homes: pd.DataFrame):
        structured, categorical = make_raw_features(homes, self.track, use_remarks=True)
        numeric = structured.drop(columns=categorical).replace([np.inf, -np.inf], np.nan)
        return homes.remarks.fillna("").astype(str), structured[categorical].astype(str), numeric

    def fit_transform(self, homes: pd.DataFrame):
        remarks, categories, numeric = self._parts(homes)
        self.numeric_columns = numeric.columns.tolist()
        self.numeric_medians = numeric.median().fillna(0.0)
        text = self.text.fit_transform(remarks)
        cats = self.one_hot.fit_transform(categories)
        nums = self.scaler.fit_transform(numeric.fillna(self.numeric_medians)).astype(np.float32)
        return sparse.hstack([text, cats, sparse.csr_matrix(nums)], format="csr", dtype=np.float32)

    def transform(self, homes: pd.DataFrame):
        remarks, categories, numeric = self._parts(homes)
        numeric = numeric.reindex(columns=self.numeric_columns).fillna(self.numeric_medians)
        text = self.text.transform(remarks)
        cats = self.one_hot.transform(categories)
        nums = self.scaler.transform(numeric).astype(np.float32)
        return sparse.hstack([text, cats, sparse.csr_matrix(nums)], format="csr", dtype=np.float32)


def target(rows: pd.DataFrame, track: str):
    if track == "list_aware":
        return np.log(rows.sold_price.to_numpy(float) / rows.list_price.to_numpy(float))
    return np.log(rows.sold_price.to_numpy(float))


def invert(raw: np.ndarray, homes: pd.DataFrame, track: str):
    if track == "list_aware":
        return homes.list_price.to_numpy(float) * np.exp(raw)
    return np.exp(raw)


def screen():
    listings = data.load("listings")
    cutoff = pd.Timestamp("2025-11-01")
    train = listings[
        (listings.date_sold < cutoff)
        & (listings.date_sold >= cutoff - pd.DateOffset(months=36))
    ].copy()
    valid = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp("2025-12-01"))].copy()
    rows = []
    prediction_frame = pd.DataFrame({"ml_num": valid.ml_num, "sold_price": valid.sold_price})
    RESULTS.mkdir(parents=True, exist_ok=True)
    for track in ("off_market", "list_aware"):
        visible = data.hide_answers(valid, track)
        for analyzer in ("word",):
            started = time.perf_counter()
            builder = SparseFeatures(track, analyzer)
            x_train = builder.fit_transform(train)
            x_valid = builder.transform(visible)
            print(track, analyzer, x_train.shape, x_train.nnz, flush=True)
            for alpha in (1.0, 10.0, 50.0, 200.0):
                model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=2000)
                model.fit(x_train, target(train, track))
                predicted = invert(model.predict(x_valid), visible, track)
                score = metrics(valid.sold_price.to_numpy(float), predicted)
                name = f"{track}_{analyzer}_ridge_a{alpha:g}"
                prediction_frame[name] = predicted
                result = {
                    "track": track,
                    "analyzer": analyzer,
                    "alpha": alpha,
                    **score,
                    "features": x_train.shape[1],
                    "seconds_cumulative": round(time.perf_counter() - started, 2),
                }
                rows.append(result)
                print(json.dumps(result), flush=True)
    frame = pd.DataFrame(rows).sort_values(["track", "mdape", "mape"])
    frame.to_csv(RESULTS / "linear_text_screen_november_2025.csv", index=False)
    prediction_frame.to_csv(RESULTS / "linear_text_screen_predictions.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    screen()
