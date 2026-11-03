"""V14 family-specific comparable learners, independent of the v13 cache."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import features
import comparable_model as v13
from ensemble_runtime import base_signature
from experiment_cache import ArrayCache, stable_key, source_digest

GROUPS = {
    "ground": ["Detached", "Semi-Detached", "Freehold Townhouse"],
    "condo": ["Condo Apt", "Condo Townhouse", "Co-Op Apt"],
}
PARAMS = dict(v13.PARAMS, random_seed=83)


def retrieve(reference, homes, family):
    """Strictly historical retrieval with family-specific similarity rules."""
    ref = reference.reset_index(drop=True); query = homes.reset_index(drop=True)
    columns = ["house_type_name", "community", "municipality"]
    rk = ref[columns].fillna("UNKNOWN").astype(str)
    qk = query[columns].fillna("UNKNOWN").astype(str)
    local = rk.groupby(columns[:2], sort=False).indices
    broad = rk.groupby([columns[0], columns[2]], sort=False).indices
    sold = ref.date_sold.to_numpy(dtype="datetime64[ns]")
    listed = query.date_listed.to_numpy(dtype="datetime64[ns]")
    r = features.off_market_features(ref); q = features.off_market_features(query)
    coords = r[["latitude", "longitude"]].to_numpy(float)
    qcoords = q[["latitude", "longitude"]].to_numpy(float)
    size = r.square_feet.to_numpy(float); qs = q.square_feet.to_numpy(float)
    rooms = r[["bedrooms", "washrooms"]].to_numpy(float)
    qr = q[["bedrooms", "washrooms"]].to_numpy(float)
    lot = (r.lot_front * r.lot_depth).to_numpy(float)
    qlot = (q.lot_front * q.lot_depth).to_numpy(float)
    condition = r[["remarks_renovated", "remarks_needs_work"]].to_numpy(float)
    qc = q[["remarks_renovated", "remarks_needs_work"]].to_numpy(float)
    styles = r.house_style.to_numpy(); qstyles = q.house_style.to_numpy()
    ids = ref.ml_num.to_numpy(); qids = query.ml_num.to_numpy()
    prices = ref.sold_price.to_numpy(float)
    valid = np.isfinite(prices) & (prices > 0) & np.isfinite(coords).all(axis=1)
    qi, ci, ds = [], [], []
    for i, (kind, community, municipality) in enumerate(qk.itertuples(index=False, name=None)):
        if np.isnat(listed[i]) or not np.isfinite(qcoords[i]).all():
            continue
        def eligible(indices):
            j = np.asarray(indices, dtype=int)
            return j[valid[j] & (ids[j] != qids[i]) & (sold[j] < listed[i])
                     & (sold[j] >= listed[i] - np.timedelta64(730, "D"))]
        j = eligible(local.get((kind, community), []))
        if len(j) < 3:
            j = eligible(broad.get((kind, municipality), []))
        if not len(j):
            continue
        distance = np.linalg.norm((coords[j] - qcoords[i]) * [111.2, 111.2 * .72], axis=1)
        age = (listed[i] - sold[j]) / np.timedelta64(1, "D")
        with np.errstate(divide="ignore", invalid="ignore"):
            size_gap = np.abs(np.log(size[j] / qs[i]))
            lot_gap = np.abs(np.log(lot[j] / qlot[i]))
        size_gap = np.where(np.isfinite(size_gap), size_gap, 1.)
        lot_gap = np.where(np.isfinite(lot_gap), lot_gap, .5)
        room_gap = np.nan_to_num(np.abs(rooms[j] - qr[i]), nan=1.).sum(axis=1)
        style_gap = (styles[j] != qstyles[i]).astype(float)
        condition_gap = np.abs(condition[j] - qc[i]).sum(axis=1)
        if family == "condo":
            # Nearby coordinates are only a proximity proxy, not a building ID.
            score = 1.5 * distance + 3 * size_gap + .5 * room_gap + .3 * style_gap + .25 * age / 365
            k = 12
        else:
            score = distance + 2 * size_gap + .6 * room_gap + .8 * style_gap + .8 * lot_gap + .5 * condition_gap + .5 * age / 365
            k = 8
        order = np.lexsort((ids[j].astype(str), score))[:k]
        qi.extend([i] * len(order)); ci.extend(j[order]); ds.extend(distance[order])
    return dict(q=np.asarray(qi, dtype=np.int64), c=np.asarray(ci, dtype=np.int64),
                distance=np.asarray(ds, dtype=np.float32))


def empty_predictions(n):
    return dict(price=np.full(n, np.nan), count=np.zeros(n, dtype=int),
                spread=np.full(n, np.nan), nearest=np.full(n, np.nan), age=np.full(n, np.nan))


class Specialist(v13.AdjustedComparables):
    def __init__(self, track, family):
        super().__init__(track)
        self.family = family

    def _house_features(self, homes, fit=False):
        X = super()._house_features(homes, fit=fit)
        X["log_size"] = np.log(X.square_feet.where(X.square_feet > 0))
        lot = X.lot_front * X.lot_depth
        X["log_lot_area"] = np.log(lot.where(lot > 0))
        return X

    def fit(self, train):
        self.reference = train.reset_index(drop=True).copy()
        if self.reference.ml_num.duplicated().any():
            raise ValueError("Specialist requires unique listing IDs")
        price = self.reference.sold_price.to_numpy(float)
        if not np.all(np.isfinite(price) & (price > 0)):
            raise ValueError("Specialist requires positive sale prices")
        self.anchors = np.log(price)
        if self.track == "list_aware":
            asking = self.reference.list_price.to_numpy(float)
            if not np.all(np.isfinite(asking) & (asking > 0)):
                raise ValueError("Specialist requires positive asking prices")
            self.anchors -= np.log(asking)
        self.model = None
        pairs = retrieve(self.reference, self.reference, self.family)
        if len(pairs["q"]) < 100:
            return self
        self.reference_features = self._house_features(self.reference, fit=True)
        X = self._pair_features(self.reference, self.reference_features, pairs)
        counts = np.bincount(pairs["q"], minlength=len(train))
        y = self.anchors[pairs["q"]] - self.anchors[pairs["c"]]
        categorical = [col for col in X if not pd.api.types.is_numeric_dtype(X[col])]
        print(f"Specialist {self.track}/{self.family}: {len(X):,} pairs from {np.count_nonzero(counts):,} homes", flush=True)
        self.model = CatBoostRegressor(**PARAMS)
        self.model.fit(X, y, cat_features=categorical, sample_weight=1. / counts[pairs["q"]])
        return self

    def predict(self, homes):
        homes = homes.reset_index(drop=True)
        result = empty_predictions(len(homes))
        if self.model is None or homes.empty:
            return result
        pairs = retrieve(self.reference, homes, self.family)
        if not len(pairs["q"]):
            return result
        X = self._pair_features(homes, self._house_features(homes), pairs)
        adjusted = self.anchors[pairs["c"]] + self.model.predict(X)
        if self.track == "list_aware":
            asking = homes.list_price.to_numpy(float)
            if not np.all(np.isfinite(asking) & (asking > 0)):
                raise ValueError("Specialist requires positive asking prices")
            adjusted += np.log(asking[pairs["q"]])
        ages = X.sale_age_days.to_numpy()
        bounds = np.r_[0, np.flatnonzero(np.diff(pairs["q"])) + 1, len(adjusted)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            i = pairs["q"][start]; values = adjusted[start:end]
            weights = np.exp(-pairs["distance"][start:end] / 2 - ages[start:end] / 365)
            if not np.isfinite(values).all() or weights.sum() <= 0:
                continue
            order = np.argsort(values, kind="stable")
            median = values[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]]
            result["price"][i] = np.exp(median)
            result["count"][i] = end - start
            result["spread"][i] = np.sqrt(np.average((values - median) ** 2, weights=weights))
            result["nearest"][i] = pairs["distance"][start:end].min()
            result["age"][i] = np.average(ages[start:end], weights=weights)
        return result


def forecast(track, train, homes):
    """Cache each family independently so completed fits survive interruptions."""
    result = empty_predictions(len(homes)); cache = ArrayCache()
    signature = source_digest([Path(__file__), Path(v13.__file__), Path(features.__file__)])
    for family, types in GROUPS.items():
        subset = train[train.house_type_name.isin(types)]
        positions = np.flatnonzero(homes.house_type_name.isin(types).to_numpy())
        if subset.empty or not len(positions):
            continue
        query = homes.iloc[positions]
        key = stable_key("specialist_pairs", dict(track=track, family=family,
                         source=signature, libraries_and_base=base_signature()), [subset, query])
        saved = cache.load("specialist_pairs", key)
        if saved is None or any(k not in saved or saved[k].shape != (len(query),) for k in result):
            print(f"Specialist {track}/{family}: training", flush=True)
            learner = Specialist(track, family).fit(subset)
            saved = learner.predict(query)
            cache.save("specialist_pairs", key, saved)
            del learner
        else:
            print(f"Specialist {track}/{family}: disk cache hit", flush=True)
        for key in result:
            result[key][positions] = saved[key]
    return result
