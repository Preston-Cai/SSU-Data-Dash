"""V13: supervised adjustments between a home and strictly earlier sales.

This is an independent challenger; the cached v10 ensemble is left intact.
No query outcomes or post-sale fields enter retrieval or pair features.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import data
import features
from ensemble_runtime import CachedEnsemble, base_signature
from experiment_cache import ArrayCache, stable_key, source_digest

K = 8
WINDOW_DAYS = 730
BLEND = {"off_market": 0.30, "list_aware": 0.20}
PARAMS = dict(iterations=2400, depth=8, learning_rate=0.04, l2_leaf_reg=12,
              loss_function="RMSE", random_seed=71, task_type="GPU", devices="0",
              verbose=200, allow_writing_files=False)


def retrieve_pairs(reference, homes):
    """Return positional query/reference indices and distances, never self sales.

    Prefer the same community and type; use municipality/type when fewer than
    three eligible community sales exist. Ranking uses only observable house
    differences and the comparable's age, not the query's outcome.
    """
    ref = reference.reset_index(drop=True)
    query = homes.reset_index(drop=True)
    keys = ["house_type_name", "community", "municipality"]
    ref_keys = ref[keys].fillna("UNKNOWN").astype(str)
    query_keys = query[keys].fillna("UNKNOWN").astype(str)
    local = ref_keys.groupby(["house_type_name", "community"], sort=False).indices
    broad = ref_keys.groupby(["house_type_name", "municipality"], sort=False).indices
    sold = pd.to_datetime(ref.date_sold).to_numpy(dtype="datetime64[ns]")
    listed = pd.to_datetime(query.date_listed).to_numpy(dtype="datetime64[ns]")
    rsize = features.square_feet(ref.house_area).to_numpy(float)
    qsize = features.square_feet(query.house_area).to_numpy(float)
    rxy = ref[["latitude", "longitude"]].to_numpy(float)
    qxy = query[["latitude", "longitude"]].to_numpy(float)
    rb = ref[["bedrooms", "washrooms"]].to_numpy(float)
    qb = query[["bedrooms", "washrooms"]].to_numpy(float)
    ids = ref.ml_num.to_numpy(); query_ids = query.ml_num.to_numpy()
    price = ref.sold_price.to_numpy(float)
    valid = np.isfinite(price) & (price > 0) & np.isfinite(rxy).all(axis=1)
    qrows, crows, distances = [], [], []
    for i, (kind, community, municipality) in enumerate(query_keys.itertuples(index=False, name=None)):
        if np.isnat(listed[i]) or not np.isfinite(qxy[i]).all():
            continue

        def eligible(indices):
            indices = np.asarray(indices, dtype=int)
            return indices[valid[indices] & (ids[indices] != query_ids[i])
                           & (sold[indices] < listed[i])
                           & (sold[indices] >= listed[i] - np.timedelta64(WINDOW_DAYS, "D"))]

        candidates = eligible(local.get((kind, community), []))
        if len(candidates) < 3:
            candidates = eligible(broad.get((kind, municipality), []))
        if not len(candidates):
            continue
        delta = (rxy[candidates] - qxy[i]) * [111.2, 111.2 * 0.72]
        distance = np.linalg.norm(delta, axis=1)
        age = (listed[i] - sold[candidates]) / np.timedelta64(1, "D")
        with np.errstate(divide="ignore", invalid="ignore"):
            size_gap = np.abs(np.log(rsize[candidates] / qsize[i]))
        size_gap = np.where(np.isfinite(size_gap), size_gap, 1.0)
        room_gap = np.nan_to_num(np.abs(rb[candidates] - qb[i]), nan=1.0).sum(axis=1)
        score = distance + 2.0 * size_gap + 0.4 * room_gap + 0.5 * age / 365
        order = np.lexsort((ids[candidates].astype(str), score))[:K]
        qrows.extend([i] * len(order)); crows.extend(candidates[order]); distances.extend(distance[order])
    return {"q": np.asarray(qrows, dtype=np.int64), "c": np.asarray(crows, dtype=np.int64),
            "distance": np.asarray(distances, dtype=np.float32)}


class AdjustedComparables:
    def __init__(self, track):
        self.track = track

    def _house_features(self, homes, fit=False):
        # Whitelist through the existing pre-sale feature function.
        safe = data.hide_answers(homes, self.track)
        X = (features.off_market_features(safe) if self.track == "off_market"
             else features.list_aware_features(safe)).reset_index(drop=True)
        X["property_type"] = homes.house_type_name.fillna("UNKNOWN").astype(str).to_numpy()
        text = homes.remarks.fillna("").astype(str)
        if fit:
            self.vectorizer = TfidfVectorizer(min_df=5, max_df=0.995, max_features=16000,
                                            ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
            self.svd = None
            try:
                sparse = self.vectorizer.fit_transform(text)
            except ValueError as error:
                if "vocabulary" not in str(error) and "terms remain" not in str(error):
                    raise
            else:
                n = min(32, sparse.shape[0] - 1, sparse.shape[1] - 1)
                if n > 0:
                    self.svd = TruncatedSVD(n_components=n, random_state=71)
                    dense = self.svd.fit_transform(sparse)
        elif self.svd is not None:
            dense = self.svd.transform(self.vectorizer.transform(text))
        if self.svd is not None:
            for j in range(dense.shape[1]):
                X[f"text_{j}"] = dense[:, j]
        return X

    def _pair_features(self, homes, X, pairs):
        q, c = pairs["q"], pairs["c"]
        left = X.iloc[q].reset_index(drop=True)
        right = self.reference_features.iloc[c].reset_index(drop=True)
        columns = {}
        for name in X:
            if pd.api.types.is_numeric_dtype(X[name]):
                a = left[name].to_numpy(dtype=np.float32)
                b = right[name].to_numpy(dtype=np.float32)
                columns[f"home_{name}"] = a
                columns[f"delta_{name}"] = a - b
            else:
                columns[f"home_{name}"] = left[name].fillna("UNKNOWN").astype(str).to_numpy()
                columns[f"comp_{name}"] = right[name].fillna("UNKNOWN").astype(str).to_numpy()
        columns["distance_km"] = pairs["distance"]
        columns["sale_age_days"] = ((homes.date_listed.to_numpy(dtype="datetime64[ns]")[q]
                                    - self.reference.date_sold.to_numpy(dtype="datetime64[ns]")[c])
                                   / np.timedelta64(1, "D")).astype(np.float32)
        # Keep the known comparable outcome outside the learner as an anchor.
        # Supplying it as an input invites learning to cancel that anchor and
        # collapse back to a subject-only price predictor.
        return pd.DataFrame(columns)

    def fit(self, train):
        self.reference = train.reset_index(drop=True).copy()
        if self.reference.ml_num.duplicated().any():
            raise ValueError("Comparable training requires unique listing IDs")
        price = self.reference.sold_price.to_numpy(float)
        if not np.all(np.isfinite(price) & (price > 0)):
            raise ValueError("Comparable training prices must be finite and positive")
        self.anchors = np.log(price)
        if self.track == "list_aware":
            asking = self.reference.list_price.to_numpy(float)
            if not np.all(np.isfinite(asking) & (asking > 0)):
                raise ValueError("List-aware training requires positive list prices")
            self.anchors -= np.log(asking)
        self.reference_features = self._house_features(self.reference, fit=True)
        pairs = retrieve_pairs(self.reference, self.reference)
        self.model = None
        if len(pairs["q"]) < 100:
            print("Comparable learner: too few historical pairs; retaining ensemble", flush=True)
            return self
        X = self._pair_features(self.reference, self.reference_features, pairs)
        y = self.anchors[pairs["q"]] - self.anchors[pairs["c"]]
        # Every subject gets equal total training weight, even with fewer comps.
        counts = np.bincount(pairs["q"], minlength=len(train))
        weights = 1.0 / counts[pairs["q"]]
        categorical = [name for name in X if not pd.api.types.is_numeric_dtype(X[name])]
        print(f"Comparable learner {self.track}: {len(X):,} pairs from "
              f"{np.count_nonzero(counts):,} homes, {len(X.columns)} features", flush=True)
        self.model = CatBoostRegressor(**PARAMS)
        self.model.fit(X, y, cat_features=categorical, sample_weight=weights)
        return self

    def predict(self, homes):
        homes = homes.reset_index(drop=True)
        result = {"price": np.full(len(homes), np.nan), "count": np.zeros(len(homes), dtype=int),
                  "spread": np.full(len(homes), np.nan), "nearest": np.full(len(homes), np.nan)}
        if self.model is None or homes.empty:
            return result
        X = self._house_features(homes)
        pairs = retrieve_pairs(self.reference, homes)
        if not len(pairs["q"]):
            return result
        table = self._pair_features(homes, X, pairs)
        adjusted = self.anchors[pairs["c"]] + self.model.predict(table)
        if self.track == "list_aware":
            asking = homes.list_price.to_numpy(float)
            if not np.all(np.isfinite(asking) & (asking > 0)):
                raise ValueError("List-aware queries require positive list prices")
            adjusted += np.log(asking[pairs["q"]])
        # Weighted median reduces the effect of one unusual comparable sale.
        boundaries = np.r_[0, np.flatnonzero(np.diff(pairs["q"])) + 1, len(adjusted)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            i = pairs["q"][start]; values = adjusted[start:end]
            weights = np.exp(-pairs["distance"][start:end] / 2.0
                             - table.sale_age_days.to_numpy()[start:end] / 365.0)
            if not np.isfinite(values).all() or weights.sum() <= 0:
                continue
            order = np.argsort(values, kind="stable")
            median = values[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]]
            result["price"][i] = np.exp(median)
            result["count"][i] = end - start
            result["spread"][i] = np.sqrt(np.average((values - median) ** 2, weights=weights))
            result["nearest"][i] = pairs["distance"][start:end].min()
        return result


class ComparableEnsemble:
    """Pairwise challenger + retained ensemble; each is reported separately."""

    def __init__(self, track):
        self.track = track

    def fit(self, train):
        self.train = train.copy(deep=True)
        self.base = CachedEnsemble(self.track).fit(train)
        return self

    def predict(self, homes):
        base = self.base.predict(homes)
        cache = ArrayCache()
        key = stable_key("adjusted_comparables", {
            "track": self.track, "base_signature": base_signature(),
            "source": source_digest([Path(__file__), Path(features.__file__)]),
        }, [self.train, homes])
        result = cache.load("adjusted_comparables", key)
        if result is None or any(name not in result or result[name].shape != (len(homes),)
                                 for name in ["price", "count", "spread", "nearest"]):
            print(f"Comparable learner {self.track}: training", flush=True)
            learner = AdjustedComparables(self.track).fit(self.train)
            result = learner.predict(homes)
            cache.save("adjusted_comparables", key, result)
        else:
            print(f"Comparable learner {self.track}: disk cache hit", flush=True)
        usable = (np.isfinite(result["price"]) & (result["price"] > 0)
                  & (result["count"] >= 3) & (result["spread"] <= 0.30) & (result["nearest"] <= 5.0))
        candidate = np.where(usable, result["price"], base)
        weight = BLEND[self.track] * usable
        delta = np.clip(np.log(candidate / base), -0.40, 0.40)
        self.component_predictions = dict(self.base.component_predictions)
        self.component_predictions["adjusted_comparables"] = candidate
        self.comparable_count = result["count"]
        self.comparable_weight = weight
        self.comparable_spread = result["spread"]
        print(f"Comparable learner {self.track}: active for {usable.sum():,}/{len(homes):,} homes", flush=True)
        return base * np.exp(weight * delta)
